from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.utils import smart_cond
from tensorflow.core.protobuf import saver_pb2
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from tensorflow.python.ops import array_ops
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from tensorflow.contrib.bayesflow import stochastic_graph as sg
from tensorflow.contrib.bayesflow import stochastic_gradient_estimators as sge

import logging
import numpy as np
import pickle as pkl
from sssp.models.model_base import ModelBase
from sssp.utils.utils import res_to_string
from sssp.models.layers.gru import GRU
from sssp.models.layers.lstm import LSTM
from sssp.models.layers.gated_lstm import GatedLSTM
from sssp.models.layers.gated_gru_multi import MultiGatedGRUCell

logging.basicConfig(level=logging.INFO)

class SemiClassifier(ModelBase):
    def __init__(self, args):
        super(SemiClassifier, self).__init__()
        self._logger = logging.getLogger(__name__)

    def _get_rnn_cell(self, rnn_type, num_units, num_layers):
        if rnn_type == 'LSTM':
            # use concated state for convinience
            cell = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=False)
        elif rnn_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units)
        else:
            raise 'The rnn type is not supported.'

        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        return cell

    def _create_placeholders(self, args):
        self.inp_l_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='inp_l_plh')

        self.tgt_l_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='tgt_l_plh')

        self.msk_l_plh = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name='msk_l_plh')

        self.inp_u_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='inp_u_plh')

        self.tgt_u_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='tgt_u_plh')

        self.msk_u_plh = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name='msk_u_plh')

        self.is_training = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name='is_training_plh')

        self.keep_prob = tf.placeholder(
                dtype=tf.float32,
                shape=[], 
                name='keep_prob_plh')

        self.beam_size_plh = tf.placeholder(
                tf.int32,
                shape=[],
                name='beam_size_plh')
        
        self.list_label_plh = []
        for i in range(len(args.task_ids.split(','))):
            self.list_label_plh.append(tf.placeholder(
                dtype=tf.int32,
                shape=[None],
                name='label_plh_{}'.format(i)))
      
    def _create_softmax_layer(self, proj, dec_outs, targets, weights, scope_name, args):
        with tf.variable_scope(scope_name):
            w_t, b = proj

            # is_training = Flase
            def get_llh_test():
                dec_outs_flt = tf.reshape(dec_outs, [-1, args.num_units])
                logits_flt = tf.add(tf.matmul(dec_outs_flt, w_t, transpose_b=True), b[None, :])
                logits = tf.reshape(logits_flt, [tf.shape(dec_outs)[0], -1, args.vocab_size])
    
                llh_precise = tf.contrib.seq2seq.sequence_loss(
                        logits=logits,
                        targets=targets,
                        weights=weights,
                        average_across_timesteps=False,
                        average_across_batch=False,
                        softmax_loss_function=None)
                return llh_precise
            
            # is_training = True
            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # use 32bit float to aviod numerical instabilites
                #w_t = tf.transpose(w)
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.nn.sampled_softmax_loss(
                        weights=local_w_t, 
                        biases=local_b,
                        inputs=local_inputs, 
                        labels=labels, 
                        num_sampled=args.num_samples,
                        num_classes=args.vocab_size,
                        partition_strategy="div")
                
            # is_training = False
            def get_llh_train():
                # if use sampled_softmax
                if args.use_sampled_softmax and args.num_samples > 0 and args.num_samples < args.vocab_size:
                    llh_train = tf.contrib.seq2seq.sequence_loss(
                            logits=dec_outs,
                            targets=targets,
                            weights=weights,
                            average_across_timesteps=False,
                            average_across_batch=False,
                            softmax_loss_function=sampled_loss)
                    self._logger.info('Use sampled softmax during training')
                else:
                    llh_train = get_llh_test()
                    self._logger.info('Use precise softmax during training')
                return llh_train
    
            loss = smart_cond(self.is_training, get_llh_train, get_llh_test)
        return loss

    def _create_embedding_matrix(self, args):
        if args.embd_path is None:
            np.random.seed(1234567890)
            embd_init = np.random.randn(args.vocab_size, args.embd_dim).astype(np.float32) * 1e-3
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

    def _create_encoder(self, inp, seqlen, scope_name, args):
        with tf.variable_scope(scope_name):
            emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)

            cell = self._get_rnn_cell(args.rnn_type, args.num_units, args.num_layers)
            hidden_states, enc_state = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=emb_inp,
                    dtype=tf.float32,
                    sequence_length=seqlen)

        return hidden_states, enc_state

    def _create_decoder(self, inp, seqlen, init_state, label_oh, weights, scope_name, args):
        with tf.variable_scope(scope_name):
            emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
            label_oh = tf.tile(label_oh[:, None, :], [1, tf.shape(emb_inp)[1], 1])
            emb_inp = tf.concat([emb_inp, label_oh], axis=2)

            cell = self._get_rnn_cell(args.rnn_type, args.num_units, args.num_layers)

            dec_outs, _ = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=emb_inp,
                    sequence_length=seqlen, 
                    initial_state=init_state)

            w_t = tf.get_variable(
                    'proj_w', 
                    [args.vocab_size, args.num_units])
            b = tf.get_variable(
                    "proj_b", 
                    [args.vocab_size])
            out_proj = (w_t, b)

        def dec_step_func(emb_t, hidden_state):
            with tf.variable_scope(scope_name):
                with tf.variable_scope('rnn', reuse=True):
                    out, state = cell(emb_t, hidden_state)
            logits = tf.add(tf.matmul(out, w_t, transpose_b=True), b[None, :])
            prob = tf.nn.log_softmax(logits)
            return state, prob

        return dec_outs, out_proj, dec_step_func, cell

    def _create_rnn_classifier(self, inp, msk, scope_name, args):
        with tf.variable_scope(scope_name):
            """
            enc_states, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=tf.contrib.rnn.GRUCell(args.num_units/2),
                    cell_bw=tf.contrib.rnn.GRUCell(args.num_units/2),
                    inputs=inp,
                    dtype=tf.float32,
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),)
            """
            inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
            init = tf.random_normal_initializer(stddev=0.1)
            key_size = 20
            mem_size = 100
            self.keys = [tf.get_variable("key_%d" % i, [key_size], initializer=init) 
                    for i in range(len(args.task_ids.split(',')))]
            cell = MultiGatedGRUCell(input_size=inp.shape[2],
                    num_units=mem_size,
                    num_tasks=len(args.task_ids.split(',')),
                    key_size=key_size,
                    keys=self.keys)
            states, final_state = tf.nn.dynamic_rnn(cell=cell,
                    inputs=inp,
                    dtype=tf.float32,
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
            
        list_logits = []
        list_weights = []
        for i in range(len(args.task_ids.split(','))):
            task_id = int(args.task_ids.split(',')[i])
            logits_i = tf.contrib.layers.fully_connected(
                    inputs=final_state[i][0],
                    num_outputs=[int(n) for n in args.num_classes.split(',')][task_id],
                    activation_fn=None,
                    scope='fc_task_{}'.format(i))
            list_logits.append(logits_i)
            list_weights.append(final_state[i][1])
        return list_logits, list_weights

    def _get_elbo_label(self, inp, tgt, msk, list_label, list_weights, args):
        """ Build encoder and decoders """
        xlen = tf.to_int32(tf.reduce_sum(msk, axis=1))
        _, enc_state = self._create_encoder(
                tgt,
                seqlen=xlen,
                scope_name='enc',
                args=args)
        
        list_num_classes = [int(n) for n in args.num_classes.split(',')]
        list_label_oh = [tf.gather(tf.eye(list_num_classes[i]), list_label[i]) for i in range(len(args.task_ids.split(',')))]
        label_oh = tf.concat(list_label_oh, axis=1)
        with tf.variable_scope('latent'):
            y_enc_in = tf.contrib.layers.fully_connected(label_oh, args.dim_z, scope='y_enc_in')
            pst_in = tf.concat([y_enc_in, enc_state], axis=1)
            mu_pst = tf.contrib.layers.fully_connected(pst_in, args.dim_z, tf.nn.tanh,
                    scope='mu_posterior')
            logvar_pst = tf.contrib.layers.fully_connected(pst_in, args.dim_z, tf.nn.tanh,
                    scope='logvar_posterior')
            mu_pri = tf.zeros_like(mu_pst)
            logvar_pri = tf.ones_like(logvar_pst)
            dist_pri = tf.contrib.distributions.Normal(mu=mu_pri, sigma=tf.exp(logvar_pri))
            dist_pst = tf.contrib.distributions.Normal(mu=mu_pst, sigma=tf.exp(logvar_pst))
            kl_loss = tf.contrib.distributions.kl(dist_pst, dist_pri)
            kl_loss = tf.reduce_sum(kl_loss, axis=1)

        with st.value_type(st.SampleValue(stop_gradient=False)):
            z_st_pri = st.StochasticTensor(dist_pri, name='z_pri')
            z_st_pst = st.StochasticTensor(dist_pst, name='z_pst')
            z = smart_cond(self.is_training, lambda: z_st_pst, lambda: z_st_pri)
       
        z_ext = tf.contrib.layers.fully_connected(tf.reshape(z, [-1, args.dim_z]), args.num_units, scope='extend_z')
        xlen = tf.to_int32(tf.reduce_sum(msk, axis=1))
        outs, proj, dec_func, cell  = self._create_decoder(
                inp,
                seqlen=xlen,
                label_oh=label_oh,
                weights=list_weights,
                init_state=z_ext,
                scope_name='dec',
                args=args)

        # build loss layers
        recons_loss = self._create_softmax_layer(
                proj=proj,
                dec_outs=outs,
                targets=tgt,
                weights=msk,
                scope_name='loss',
                args=args)
        
        return recons_loss, kl_loss
    
    def get_loss_l(self, args):
        with tf.variable_scope(args.log_prefix):
            """ label CLASSIFICATION """
            self.list_logits_l, self.list_weights_l = self._create_rnn_classifier(self.tgt_l_plh,
                    self.msk_l_plh,
                    scope_name='clf',
                    args=args)
            self.list_predict_loss_l = []
            self.list_accucary_l = []
            for i in range(len(args.task_ids.split(','))):
                predict_loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.list_label_plh[i],
                    logits=self.list_logits_l[i])
                predict_loss_i = tf.reduce_mean(predict_loss_i)
                acc_i = tf.equal(tf.cast(self.list_label_plh[i], tf.int64), tf.argmax(self.list_logits_l[i], axis=1))
                acc_i = tf.reduce_mean(tf.cast(acc_i, tf.float32))
                self.list_predict_loss_l.append(predict_loss_i)
                self.list_accucary_l.append(acc_i)
 
            self.predict_loss_l = tf.add_n(self.list_predict_loss_l) / len(args.task_ids.split(','))
            self.accuracy_l = tf.add_n(self.list_accucary_l) / len(args.task_ids.split(','))

            """ label CVAE """
            self.recons_loss_l, self.kl_loss_l = self._get_elbo_label(self.inp_l_plh,
                    self.tgt_l_plh,
                    self.msk_l_plh,
                    self.list_label_plh,
                    self.list_weights_l,
                    args)

            #DEBUG
            #self.recons_loss_l = tf.reduce_sum(self.recons_loss_l * tf.stop_gradient(self.weights_l), axis=1)
            self.recons_loss_l = tf.reduce_sum(self.recons_loss_l * self.msk_l_plh/tf.reduce_sum(self.msk_l_plh, axis=1, keep_dims=True), axis=1)

            self.recons_loss_l = tf.reduce_mean(self.recons_loss_l)
            self.ppl_l = tf.exp(self.recons_loss_l)
            self.kl_loss_l = tf.reduce_mean(self.kl_loss_l)
            self.elbo_loss_l = self.recons_loss_l + self.kl_loss_l * self.kl_w

            self.loss_l = self.elbo_loss_l + self.predict_loss_l * args.alpha

            tf.summary.scalar('elbo_loss_l', self.elbo_loss_l)
            tf.summary.scalar('kl_w', self.kl_w)
            tf.summary.scalar('ppl_l', self.ppl_l)
            tf.summary.scalar('kl_loss_l', self.kl_loss_l)
            tf.summary.scalar('pred_loss_l', self.predict_loss_l)
            tf.summary.scalar('accuracy_l', self.accuracy_l)
        return self.loss_l
    
    def get_loss_u(self, args):
        with tf.variable_scope(args.log_prefix, reuse=True):
            """ unlabel CLASSIFICATION """
            self.list_logits_u, self.list_weights_u = self._create_rnn_classifier(self.tgt_u_plh,
                    self.msk_u_plh,
                    scope_name='clf',
                    args=args)
            self.list_predict_u = []
            self.list_entropy_u = []
            for i in range(len(args.task_ids.split(','))):
                predict_i = tf.nn.softmax(self.list_logits_u[i])
                entropy_i = tf.losses.softmax_cross_entropy(predict_i, predict_i)
                self.list_predict_u.append(predict_i)
                self.list_entropy_u.append(entropy_i)
            self.entropy_u = tf.add_n(self.list_entropy_u) / len(args.task_ids.split(','))

            """ unlabel CVAE """
            with st.value_type(st.SampleValue(stop_gradient=True)):
                list_y_st = []
                for i in range(len(args.task_ids.split(','))):
                    list_y_st.append(st.StochasticTensor(tf.contrib.distributions.Categorical(p=self.list_predict_u[i]), 
                        name='dist_y_{}'.format(i),
                        ))#loss_fn=sge.get_score_function_with_baseline()))

                recons_loss_u_s, kl_loss_u_s = self._get_elbo_label(self.inp_u_plh,
                        self.tgt_u_plh,
                        self.msk_u_plh,
                        list_y_st,
                        self.list_weights_u,
                        args)

                recons_loss_u_s = tf.reduce_sum(recons_loss_u_s * self.msk_u_plh/tf.reduce_sum(self.msk_u_plh, axis=1, keep_dims=True), axis=1)
                recons_loss_u_s = tf.reduce_mean(recons_loss_u_s)
                routing_loss = recons_loss_u_s + kl_loss_u_s * self.kl_w

                list_baseline = []
                zeros = tf.zeros([tf.shape(self.inp_u_plh)[0]], dtype=tf.int64)
                for i in range(len(args.task_ids.split(','))):
                    recons_loss_tmp, kl_loss_tmp = self._get_elbo_label(self.inp_u_plh,
                            self.tgt_u_plh,
                            self.msk_u_plh,
                            list_y_st[:i] + [zeros] + list_y_st[i+1:],
                            self.list_weights_u,
                            args)
                    recons_loss_tmp = tf.reduce_sum(recons_loss_tmp * self.msk_u_plh/tf.reduce_sum(self.msk_u_plh, axis=1, keep_dims=True), axis=1)
                    recons_loss_tmp = tf.reduce_mean(recons_loss_tmp)
                    routing_loss_tmp = recons_loss_tmp + kl_loss_tmp * self.kl_w
                    list_baseline.append(routing_loss_tmp)

        with tf.variable_scope(args.log_prefix, reuse=False):
            list_surrogate_loss = [list_y_st[i].loss(routing_loss-list_baseline[i]) for i in range(len(args.task_ids.split(',')))]
            surrogate_loss = tf.add_n(list_surrogate_loss) / len(list_surrogate_loss)
        
        return tf.reduce_mean(surrogate_loss) + tf.reduce_mean(routing_loss) + self.entropy_u

    def model_setup(self, args):
        with tf.variable_scope(args.log_prefix):
            self.init_global_step()
            self._create_placeholders(args)
            self._logger.info("Created placeholders.")
            self._create_embedding_matrix(args)

            self.kl_w = tf.log(1. + tf.exp((self.global_step - args.klw_b) * args.klw_w))
            self.kl_w = tf.minimum(self.kl_w, 1.) / 100.0 #scale reweighted
        
        self.loss_l = self.get_loss_l(args)
        self.train_unlabel = tf.greater(self.global_step, args.num_pretrain_steps)
        #self.loss_u = smart_cond(self.train_unlabel, lambda: self.get_loss_u(args), lambda: tf.constant(0.))
        self.loss_u = self.get_loss_u(args)
        tf.summary.scalar('train_unlabel', tf.to_int64(self.train_unlabel))
        tf.summary.scalar('loss_u', self.loss_u)

        self.loss = self.loss_l + self.loss_u
        tf.summary.scalar('loss', self.loss)

        with tf.variable_scope(args.log_prefix):
            # optimizer
            #embd_var = self.embedding_matrix
            #other_var_list = [v for v in tf.trainable_variables() if v.name != embd_var.name]
            learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 
                    args.decay_steps,
                    args.decay_rate,
                    staircase=True)
            self.train_op = self.training_op(self.loss, #tf.trainable_variables(),
                    tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix),
                    grad_clip=args.grad_clip,
                    max_norm=args.max_norm,
                    train_embd=True,
                    learning_rate=args.learning_rate,)
            self._logger.info("Created SemiClassifier Model.")

            self._create_saver(args)
            self._logger.info('Created Saver')

            self.merged = tf.summary.merge_all()

            """ Create beam search layer
            self.beam_output_cur, self.beam_scores_cur = self._create_beam_search_layer(
                    init_state=yz,
                    dec_step_func=cur_dec_func,
                    cell=cur_cell,
                    embedding_matrix=self.embedding_matrix,
                    vocab_size=args.vocab_size,
                    num_layers=args.num_layers,)
            self._logger.info('Created Beam Search Layer')
            """

            vt = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix)
            vs = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=args.log_prefix)
        return vt, vs

    def run_batch(self, sess, inps, istrn=True):
        plhs = [self.inp_l_plh, self.tgt_l_plh, self.msk_l_plh,] + self.list_label_plh +  \
                [self.inp_u_plh, self.tgt_u_plh, self.msk_u_plh,]

        if istrn:
            fetch_dict = [['elbo_l', self.elbo_loss_l],
                    ['ppl_l', self.ppl_l],
                    ['kl_l', self.kl_loss_l],
                    ['pred_l', self.predict_loss_l],
                    ['acc_l', self.accuracy_l],
                    ['loss_u', self.loss_u],
                    ['train_u', self.train_unlabel],
                    ['klw', self.kl_w]]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training, True]])
            fetch = [self.merged] + [t[1] for t in fetch_dict] + [self.train_op]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+1]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        else:
            fetch_dict = [['pred_l', self.predict_loss_l],
                    ['acc_l', self.accuracy_l],]
            feed_dict = dict(list(zip(plhs, inps+inps[:-1])) + [[self.is_training, False]])
            fetch = [self.merged] + [t[1] for t in fetch_dict]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+1]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        return res_dict, res_str, res[0]

    def _create_saver(self, args):
        # -------------- initialization and restore ---------------
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.VARIABLES, scope=args.log_prefix),
                max_to_keep=args.max_to_keep,
                write_version=saver_pb2.SaverDef.V2)  # save all, including word embeddings
        return self.saver
    
    def get_prepare_func(self, args):
        def prepare_data(raw_inp):
            raw_inp = [[s.split(' ') for s in l.strip().split('\t')] for l in raw_inp[0]]
            raw_inp = list(zip(*raw_inp))
            labels = [raw_inp[int(i)] for i in args.task_ids.split(',')]
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
            labels = [np.asarray(l).flatten().astype('int32') for l in labels]

            return list(inp) + labels
        return prepare_data
