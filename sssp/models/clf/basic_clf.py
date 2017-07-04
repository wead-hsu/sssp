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
            
            if args.rnn_type == 'LSTM':
                cell = tf.contrib.rnn.LSTMCell(args.num_units, state_is_tuple=True, use_peepholes=True)
                if args.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                _, enc_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                if args.num_layers == 1:
                    enc_state = enc_state[-1]
                else:
                    enc_state = enc_state[-1][-1]
                weights = tf.zeros(tf.shape(emb_inp)[:2])
            elif args.rnn_type == 'GRU':
                cell = tf.contrib.rnn.GRUCell(args.num_units)
                if args.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                _, enc_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
                self.gate_weights = weights
            elif args.rnn_type == 'BiGRU':
                cell = tf.contrib.rnn.GRUCell(args.num_units/ 2)
                _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, 
                        cell_bw=cell, 
                        inputs=emb_inp,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),
                        dtype=tf.float32)
                enc_state = tf.concat(enc_state, axis=1)
                weights = tf.zeros(tf.shape(emb_inp)[:2])
                self.gate_weights = weights
            elif args.rnn_type == 'BiGRU+maxpooling':
                cell = tf.contrib.rnn.GRUCell(args.num_units/ 2)
                enc_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, 
                        cell_bw=cell, 
                        inputs=emb_inp,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),
                        dtype=tf.float32)
                enc_state = tf.concat(enc_states, axis=2)
                enc_state = tf.reduce_max(enc_state, axis=1)
                weights = tf.zeros(tf.shape(emb_inp)[:2])
                self.gate_weights = weights
            elif args.rnn_type == 'GatedGRU':
                from sssp.models.layers.gated_gru import GatedGRU
                """
                enc_layer = GatedGRU(emb_inp.shape[2], args.num_units)
                enc_state, weights = enc_layer.forward(emb_inp, msk, return_final=True)
                """
                cell = GatedGRU(emb_inp.shape[2], args.num_units)
                enc_states, enc_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        initial_state=cell.zero_state(tf.shape(emb_inp)[0], dtype=tf.float32),
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                enc_state = enc_state[0]
                weights = enc_states[1]
                self.gate_weights = weights
                self._logger.debug(enc_state.shape)
                self._logger.debug(weights.shape)
            elif args.rnn_type == 'GatedCtxGRU':
                from sssp.models.layers.gated_gru_with_context import GatedGRU

                inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
                def _reverse(input_, seq_lengths, seq_dim, batch_dim):
                    if seq_lengths is not None:
                        return array_ops.reverse_sequence(
                            input=input_, seq_lengths=seq_lengths,
                            seq_dim=seq_dim, batch_dim=batch_dim)
                    else:
                        return array_ops.reverse(input_, axis=[seq_dim])
                
                sequence_length = tf.to_int64(tf.reduce_sum(msk, axis=1))
                time_dim = 1
                batch_dim = 0
                cell_bw = tf.contrib.rnn.GRUCell(args.num_units)
                with tf.variable_scope("bw") as bw_scope:
                  inputs_reverse = _reverse(
                          inp, seq_lengths=sequence_length,
                          seq_dim=time_dim, batch_dim=batch_dim)
                  tmp, output_state_bw = tf.nn.dynamic_rnn(
                          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
                          dtype=tf.float32,
                          scope=bw_scope)
            
                output_bw = _reverse(
                        tmp, seq_lengths=sequence_length,
                        seq_dim=time_dim, batch_dim=batch_dim)
            
                enc_layer = GatedGRU(inp.shape[2], args.num_units, args.num_units)
                enc_state, weights = enc_layer.forward(inp, output_bw, msk, return_final=True)
                self.gate_weights = weights
                weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
                #weights = msk / tf.reduce_sum(msk, axis=1, keep_dims=True)
            elif args.rnn_type == 'tagGatedGRU':
                from sssp.models.layers.gated_gru_with_context import GatedGRU
                sequence_length = tf.to_int64(tf.reduce_sum(msk, axis=1))
                cell_tag = tf.contrib.rnn.GRUCell(args.num_units)
                with tf.variable_scope("tagrnn") as scope:
                    outputs, final_state = tf.nn.dynamic_rnn(
                            cell=cell_tag, inputs=emb_inp, sequence_length=sequence_length,
                            dtype=tf.float32,
                            scope=scope)
                
                #inp = tf.concat([emb_inp, outputs], axis=2)
                gatedctxgru_layer = GatedGRU(emb_inp.shape[2], args.num_units, args.num_units)
                enc_state, weights = gatedctxgru_layer.forward(emb_inp, outputs, msk, return_final=True)
                self.gate_weights = weights
                weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
            elif args.rnn_type == 'Tag=bigru;Gatedgru=bwctx+tag':
                from sssp.models.layers.gated_gru_with_context import GatedGRU
                sequence_length = tf.to_int64(tf.reduce_sum(msk, axis=1))
                
                tag_states, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=tf.contrib.rnn.GRUCell(100),
                        cell_bw=tf.contrib.rnn.GRUCell(100),
                        inputs=emb_inp,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),
                        dtype=tf.float32)
                tag_states = tf.concat(tag_states, axis=2)
        
                def _reverse(input_, seq_lengths, seq_dim, batch_dim):
                    if seq_lengths is not None:
                        return array_ops.reverse_sequence(
                            input=input_, seq_lengths=seq_lengths,
                            seq_dim=seq_dim, batch_dim=batch_dim)
                    else:
                        return array_ops.reverse(input_, axis=[seq_dim])
                
                time_dim = 1
                batch_dim = 0
                cell_bw = tf.contrib.rnn.GRUCell(args.num_units)
                with tf.variable_scope("bw") as bw_scope:
                    inputs_reverse = _reverse(
                            tf.concat([emb_inp, tag_states], axis=2), seq_lengths=sequence_length,
                            seq_dim=time_dim, batch_dim=batch_dim)
                    tmp, output_state_bw = tf.nn.dynamic_rnn(
                            cell=tf.contrib.rnn.GRUCell(100), 
                            inputs=inputs_reverse, 
                            sequence_length=sequence_length,
                            dtype=tf.float32,
                            scope=bw_scope)
            
                output_bw = _reverse(
                        tmp, seq_lengths=sequence_length,
                        seq_dim=time_dim, batch_dim=batch_dim)
            
                enc_layer = GatedGRU(
                        inp_size=emb_inp.shape[2], 
                        context_size=100+args.embd_dim, 
                        num_units=args.num_units)
                enc_state, weights = enc_layer.forward(
                        inp=emb_inp, 
                        ctx=tf.concat([output_bw, emb_inp], axis=2),
                        msk=msk, 
                        return_final=True)
                self.gate_weights = weights
                #weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
            elif args.rnn_type == 'EntityNetwork':
                from sssp.models.layers.entity_network import EntityNetwork
                cell = tf.contrib.rnn.GRUCell(args.num_units)
                if args.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                enc_states, _ = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training, args.keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))

                init = tf.random_normal_initializer(stddev=0.1)
                keys = [tf.get_variable("key_%d" % i, [args.num_units], initializer=init) for i in range(1)]
                cell = EntityNetwork(memory_slots=1,
                        memory_size=args.num_units,
                        keys=keys)
                output, final_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=enc_states,
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                enc_state = final_state[0][0]
                weights = output[1][0]
                self.gate_weights = weights
            else:
                raise 'RNN type {} not supported'.format(args.rnn_type)

            self._logger.info("Encoder done")
            return enc_state, weights

    def _create_fclayers(self, enc_state, num_classes, scope, args):
        #enc_state = tf.nn.dropout(enc_state, tf.where(self.is_training, 0.5, 1.0)) # add == worse
        enc_state = tf.contrib.layers.fully_connected(enc_state, 30, scope=scope+'fc0')
        enc_state = tf.nn.tanh(enc_state)
        #enc_state = tf.nn.softmax(enc_state) # add == slow
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
            enc_state, gate_weights = self._create_encoder(
                    inp=self.input_plh,
                    msk=self.mask_plh,
                    keep_rate=args.keep_rate,
                    scope_name='enc_rnn',
                    args=args)

            self.loss_reg_l1 = tf.reduce_mean(tf.reduce_sum(gate_weights, axis=1))
            self.loss_reg_diff = tf.reduce_sum(tf.abs(gate_weights[:, :-1, :] - gate_weights[:, 1:, :]), axis=1)
            self.loss_reg_diff = tf.reduce_mean(self.loss_reg_diff)
            self.loss_reg_sharp = tf.reduce_sum(gate_weights * (1-gate_weights), axis=1)
            self.loss_reg_sharp = tf.reduce_mean(self.loss_reg_sharp)

            if args.fixirrelevant == False:
                logits = self._create_fclayers(enc_state, args.num_classes, 'fclayers', args)
                self.prob = tf.nn.softmax(logits)
                self.pred = tf.argmax(logits, axis=1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.label_plh), tf.float32))
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label_plh, logits=logits)
                self.loss = tf.reduce_mean(self.loss)
            else:
                print('Use fixirrelevant')
                logits = self._create_fclayers(enc_state, args.num_classes-1, 'fclayers', args)
                full_logits = tf.concat([tf.zeros([batch_size,1]), logits], axis=1) # the irrelevant is concatenated on the left
                self.prob = tf.nn.softmax(full_logits)
                self.pred = tf.argmax(full_logits, axis=1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.label_plh), tf.float32))
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label_plh, logits=full_logits)
                self.loss = tf.reduce_mean(self.loss)
            
            self.loss_total = self.loss + args.w_regl1 * self.loss_reg_l1 + \
                    args.w_regdiff * self.loss_reg_diff + \
                    args.w_regsharp * self.loss_reg_sharp 
            learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 
                    args.decay_steps,
                    args.decay_rate,
                    staircase=True)
            self.train_op = self.training_op(self.loss_total, 
                    tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix),
                    grad_clip=args.grad_clip,
                    max_norm=args.max_norm,
                    train_embd=True,
                    learning_rate=args.learning_rate,)
            self._logger.info("Created RnnClassifier.")
            self._create_saver(args)
            self._logger.info('Created Saver')

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('loss_reg_l1', self.loss_reg_l1)
            tf.summary.scalar('loss_reg_diff', self.loss_reg_diff)
            tf.summary.scalar('loss_reg_sharp', self.loss_reg_sharp)
            tf.summary.scalar('loss_total', self.loss_total)
            self.merged = tf.summary.merge_all()

            vt = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix)
            vs = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=args.log_prefix)
        return vt, vs

    def run_batch(self, sess, inps, istrn=True):
        plhs = [self.input_plh,
                self.mask_plh,
                self.label_plh]

        if istrn:
            fetch_dict = [
                    ['loss_total', self.loss_total],
                    ['loss', self.loss],
                    ['loss_reg_l1', self.loss_reg_l1],
                    ['loss_reg_diff', self.loss_reg_diff],
                    ['loss_reg_sharp', self.loss_reg_sharp],
                    ['acc', self.accuracy],
                    ]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training, istrn]])
            fetch = [self.merged, self.gate_weights] + [t[1] for t in fetch_dict] + [self.train_op]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+2]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        else:
            fetch_dict = [
                    ['loss_total', self.loss_total],
                    ['loss', self.loss],
                    ['loss_reg_l1', self.loss_reg_l1],
                    ['loss_reg_diff', self.loss_reg_diff],
                    ['loss_reg_sharp', self.loss_reg_sharp],
                    ['acc', self.accuracy],
                    ]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training, istrn]])
            fetch = [self.merged, self.gate_weights] + [t[1] for t in fetch_dict]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+2]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        return res_dict, res_str, res[0], res[1]
    
    def classify(self, sess, sent, mask):
        feed_dict = {self.input_plh: sent, self.mask_plh: mask, self.is_training: False}
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
        def prepare_data(inp):
            inp = [[s.split(' ') for s in l.strip().split('\t')] for l in inp[0]]
            inp = list(zip(*inp))
            label, inp = inp
             
            def proc(sents):
                sent_lens = [len(s) for s in sents]
                max_sent_len = min(args.max_sent_len, max(sent_lens))
                batch_size = len(sents)
                inp_np = np.zeros([batch_size, max_sent_len+1], dtype='int64')
                tgt_np = np.zeros([batch_size, max_sent_len+1], dtype='int64')
                msk_np = np.zeros([batch_size, max_sent_len+1], dtype='float32')
                for idx, s in enumerate(sents):
                    len_s = min(max_sent_len, len(s))
                    inp_np[idx][1:len_s+1] = s[:len_s]
                    tgt_np[idx][:len_s] = s[:len_s]
                    msk_np[idx][:len_s+1] = 1
                return inp_np, tgt_np, msk_np
            
            inp = proc(inp)
            inp = (inp[1], inp[2])
            label = np.asarray(label).flatten().astype('int64')

            return inp + (label,)
        return prepare_data
