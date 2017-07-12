from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.utils import smart_cond
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.ops import array_ops

import logging
import numpy as np
import pickle as pkl
from sssp.models.model_base import ModelBase
from sssp.utils.utils import res_to_string
from sssp.models.layers.gru import GRU
from sssp.models.layers.lstm import LSTM
from sssp.models.layers.gated_lstm import GatedLSTM

class RnnClassifier(ModelBase):
    def __init__(self, args):
        super(RnnClassifier, self).__init__()
        self._logger = logging.getLogger(__name__)
       
    def _create_placeholders(self, args):
        self.input_plh = tf.placeholder(
                dtype=tf.int64,
                shape=[None, None],
                name='input_plh')

        self.mask_plh = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name='mask_plh')

        self.label_plh = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='label_plh')

        self.is_training = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name='is_training')

    def _create_embedding_matrix(self, args):
        if args.embd_path is None:
            np.random.seed(1234567890)
            embd_init = np.random.randn(args.vocab_size, args.embd_dim).astype(np.float32) * 1e-2
        else:
            embd_init = pkl.load(open(args.embd_path, 'rb'))
            assert embd_init.shape[0] == args.vocab_size \
                    and embd_init.shape[1] == args.embd_dim, \
                'Shapes between given pretrained embedding matrix and given settings do not match'

        with tf.variable_scope('embedding_matrix'):
            self.embedding_matrix = tf.get_variable(
                    'embedding_matrix', 
                    [args.vocab_size, args.embd_dim],
                    initializer=tf.constant_initializer(embd_init))

    def _create_encoder(self, inp, msk, keep_rate, scope_name, args):
        with tf.variable_scope(scope_name):
            emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
            
            #list_enc_state = []
            if args.rnn_type == 'GatedGRU':
                from sssp.models.layers.gated_gru import GatedGRU
                cell = GatedGRU(emb_inp.shape[2], args.num_units)
                enc_states, enc_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                weights = enc_states[1]
                self.weights = weights
                self._logger.debug(enc_state[0].shape)
                self._logger.debug(enc_state[1].shape)
                self._logger.debug(weights.shape)
            elif args.rnn_type == 'GRU':
                cell = tf.contrib.rnn.GRUCell(args.num_units)
                if args.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                _, enc_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                weights = tf.zeros(tf.shape(emb_inp)[:2])
                enc_state = [enc_state, weights]
            self._logger.info("Encoder done")
            return enc_state

    def _create_fclayers(self, enc_state, num_classes, scope, args):
        enc_state = tf.contrib.layers.fully_connected(enc_state, 30, scope=scope+'fc0')
        enc_state = tf.nn.tanh(enc_state)
        enc_state = tf.nn.dropout(enc_state, tf.where(self.is_training, 0.5, 1.0))
        
        logits = tf.contrib.layers.fully_connected(
                inputs=enc_state,
                num_outputs=num_classes,
                activation_fn=None,
                scope=scope+'fc1')

        return logits

    def model_setup(self, args):
        with tf.variable_scope(args.log_prefix):
            self.init_global_step()
            self._create_placeholders(args)
            self._create_embedding_matrix(args)
            
            batch_size = tf.shape(self.input_plh)[0]
            seqlen = tf.to_int64(tf.reduce_sum(self.mask_plh, axis=1))
            enc_state, _ = self._create_encoder(
                    inp=self.input_plh,
                    msk=self.mask_plh,
                    keep_rate=args.keep_rate,
                    scope_name='enc_rnn',
                    args=args)

            logits = self._create_fclayers(enc_state, args.num_classes-1, 'fclayers', args)
            full_logits = tf.concat([tf.zeros([batch_size,1]), logits], axis=1) # the irrelevant is concatenated on the left
            
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.label_plh, full_logits)
            self.loss = tf.reduce_mean(self.loss)
            
            # duplicate
            self.pred = tf.argmax(full_logits, axis=1)
            self.prob = tf.nn.softmax(full_logits)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.label_plh), tf.float32))
           
            tf.summary.scalar('loss', self.loss)
            
            '''
            learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 
                    args.decay_steps,
                    args.decay_rate,
                    staircase=True)
            '''
            self.train_op = self.training_op(self.loss,
                    tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix),
                    grad_clip=args.grad_clip,
                    max_norm=args.max_norm,
                    train_embd=True,
                    learning_rate=args.learning_rate,)
            self._logger.info("Created RnnClassifier.")

            self._create_saver(args)
            self._logger.info('Created Saver')

            self.merged = tf.summary.merge_all()

            vt = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix)
            vs = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=args.log_prefix)
        return vt, vs

    def run_batch(self, sess, inps, istrn=True):
        plhs = [self.input_plh,
                self.mask_plh, self.label_plh]

        if istrn:
            fetch_dict = [
                    ['loss', self.loss],
                    ['acc', self.accuracy],
                    ]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training, istrn]])
            fetch = [self.merged] + [t[1] for t in fetch_dict] + [self.train_op]
            res = sess.run(fetch, feed_dict)
            #print(sess.run(self.prob_leaves, feed_dict))
            res_dict = dict([[fetch_dict[i][0], res[i+1]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        else:
            fetch_dict = [
                    ['loss', self.loss],
                    ['acc', self.accuracy],
                    ]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training, istrn]])
            fetch = [self.merged] + [t[1] for t in fetch_dict]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+1]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        return res_dict, res_str, res[0]
    
    def classify(self, sess, sent, mask):
        feed_dict = {self.input_plh: sent, self.mask_plh: mask, self.is_training: False}
        fetch = [self.prob, self.pred]
        prob, pred = sess.run(fetch, feed_dict)
        return prob, pred

    def _create_saver(self, args):
        # -------------- initialization and restore ---------------
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=args.log_prefix),
                max_to_keep=args.max_to_keep,
                write_version=saver_pb2.SaverDef.V2)  # save all, including word embeddings
        return self.saver

    def get_prepare_func(self, args):
        def prepare_data(raw_inp):
            raw_inp = [[s.split(' ') for s in l.strip().split('\t')] for l in raw_inp[0]]
            raw_inp = list(zip(*raw_inp))
            labels = raw_inp[:-1]
            inp = raw_inp[-1]
            
            def proc(sents):
                sent_lens = [len(s) for s in sents]
                max_sent_len = max(sent_lens)
                batch_size = len(sents)
                inp_np = np.zeros([batch_size, max_sent_len+1], dtype='int64')
                tgt_np = np.zeros([batch_size, max_sent_len+1], dtype='int64')
                msk_np = np.zeros([batch_size, max_sent_len+1], dtype='float32')
                for idx, s in enumerate(sents):
                    inp_np[idx][1:len(s)+1] = s
                    tgt_np[idx][:len(s)] = s
                    msk_np[idx][:len(s)+1] = 1
                return inp_np, tgt_np, msk_np
            
            inp = proc(inp)
            inp = [inp[1], inp[2]]
            labels = [np.asarray(label).flatten().astype('int64') for label in labels]
            #print(len(inp + labels))
            #print(inp + labels)
            return inp + labels
        return prepare_data