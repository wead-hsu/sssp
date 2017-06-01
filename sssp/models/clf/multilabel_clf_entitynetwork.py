from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.utils import smart_cond
from tensorflow.core.protobuf import saver_pb2

import logging
import numpy as np
import pickle as pkl
from sssp.models.model_base import ModelBase
from sssp.utils.utils import res_to_string
from sssp.models.layers.gru import GRU
from sssp.models.layers.lstm import LSTM
from sssp.models.layers.gated_gru import GatedGRU
from sssp.models.layers.gated_lstm import GatedLSTM
from sssp.models.layers.entity_network import EntityNetwork
#from sssp.models.layers.gru_test import GRU

class MultiLabelClassifier(ModelBase):
    def __init__(self, args):
        super(MultiLabelClassifier, self).__init__()
        self._logger = logging.getLogger(__name__)
       
    def _create_placeholders(self, args):
        self.input_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='input_plh')

        self.mask_plh = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name='mask_plh')
        
        self.labels_plh = []
        for i in range(args.num_tasks):
            self.labels_plh.append(tf.placeholder(
                dtype=tf.int32,
                shape=[None],
                name='label_plh_{}'.format(i)))

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
            
            if args.rnn_type == 'LSTM':
                cell = tf.contrib.rnn.LSTMCell(args.num_units, state_is_tuple=True, use_peepholes=True)
                if args.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                enc_states, final_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                weights = tf.zeros(tf.shape(emb_inp)[:2])
            elif args.rnn_type == 'GRU':
                cell = tf.contrib.rnn.GRUCell(args.num_units)
                if args.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                enc_states, final_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                weights = tf.zeros(tf.shape(emb_inp)[:2])
            elif args.rnn_type == 'GatedGRU':
                cell = GatedGRU(emb_inp.shape[2], args.num_units)
                if args.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                enc_states, final_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                enc_states, weights = enc_states
            else:
                raise 'Rnn type not supported'

            self._logger.info("Encoder done")
            return enc_states, weights

    def _create_classifier(self, inp, msk, args):
        init = tf.random_normal_initializer(stddev=0.1)
        self.keys = [tf.get_variable("key_%d" % i, [args.num_units], initializer=init) 
                for i in range(args.num_tasks)]
        cell = EntityNetwork(memory_slots=args.num_tasks,
                memory_size=args.num_units,
                keys=self.keys)
        output, final_state = tf.nn.dynamic_rnn(cell=cell,
                inputs=inp,
                dtype=tf.float32,
                sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
        return final_state
    
    def model_setup(self, args):
        with tf.variable_scope(args.log_prefix):
            self.init_global_step()
            self._create_placeholders(args)
            self._create_embedding_matrix(args)

            seqlen = tf.to_int32(tf.reduce_sum(self.mask_plh, axis=1))
            enc_states, _ = self._create_encoder(
                    inp=self.input_plh,
                    msk=self.mask_plh,
                    keep_rate=args.keep_rate,
                    scope_name='enc_rnn',
                    args=args)

            states = self._create_classifier(
                    inp=enc_states,
                    msk=self.mask_plh,
                    args=args)
            
            self.prob_list = []
            self.accuracy_list = []
            self.loss_list = []
            for i in range(args.num_tasks):
                logits_i = tf.contrib.layers.fully_connected(
                    inputs=states[i],
                    num_outputs=[int(n) for n in args.num_classes.split(',')][i],
                    activation_fn=None,
                    scope='fc_task_{}'.format(i))
                
                prob = tf.nn.softmax(logits_i)
                acc = tf.equal(tf.cast(self.labels_plh[i], tf.int64), tf.argmax(logits_i, axis=1))
                acc = tf.reduce_mean(tf.cast(acc, tf.float32))
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels_plh[i],
                    logits=logits_i)
                loss = tf.reduce_mean(loss)
                self.prob_list.append(prob)
                self.accuracy_list.append(acc)
                self.loss_list.append(loss)
               
            self.loss = tf.add_n(self.loss_list) / args.num_tasks
            self.accuracy = tf.add_n(self.accuracy_list) / args.num_tasks

            tf.summary.scalar('loss', self.loss)

            learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 
                    args.decay_steps,
                    args.decay_rate,
                    staircase=True)
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
        plhs = [self.input_plh, self.mask_plh] + self.labels_plh

        if istrn:
            fetch_dict = [
                    ['loss', self.loss],
                    ['acc', self.accuracy],
                    ]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training, istrn]])
            fetch = [self.merged] + [t[1] for t in fetch_dict] + [self.train_op]
            res = sess.run(fetch, feed_dict)
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
        feed_dict = {self.input_plh: sent, self.mask_plh: mask}
        fetch = [self.prob]
        prob = sess.run(fetch, feed_dict)
        return prob

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
            labels = raw_inp[:args.num_tasks]
            #label = labels[args.task_id]
            inp = raw_inp[args.num_tasks]
            
            def proc(sents):
                sent_lens = [len(s) for s in sents]
                max_sent_len = min(args.max_sent_len, max(sent_lens))
                batch_size = len(sents)
                inp_np = np.zeros([batch_size, max_sent_len+1], dtype='int32')
                tgt_np = np.zeros([batch_size, max_sent_len+1], dtype='int32')
                msk_np = np.zeros([batch_size, max_sent_len+1], dtype='float32')
                for idx, s in enumerate(sents):
                    len_s = min(max_sent_len, len(s))
                    inp_np[idx][1:len_s+1] = s[:len_s]
                    tgt_np[idx][:len_s] = s[:len_s]
                    msk_np[idx][:len_s+1] = 1
                return inp_np, tgt_np, msk_np
            
            inp = proc(inp)
            inp = [inp[1], inp[2]]
            labels = [np.asarray(l).flatten().astype('int32') for l in labels]

            return inp + labels
        return prepare_data
