import tensorflow as tf
import functools

def prelu_func(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
prelu = functools.partial(prelu_func, initializer=tf.constant_initializer(1.0))

class EntityNetwork(tf.contrib.rnn.RNNCell):
    def __init__(self, memory_slots, memory_size, keys, activation=prelu,
            initializer=tf.random_normal_initializer(stddev=0.1), scope_name='etynet'):
        # intializer
        self.m, self.mem_sz, self.keys = memory_slots, memory_size, keys
        self.activation, self.init = activation, initializer

        # parameters
        with tf.variable_scope(scope_name):
            self.U = tf.get_variable('U', shape=[self.mem_sz, self.mem_sz], initializer=self.init)
            self.V = tf.get_variable('V', shape=[self.mem_sz, self.mem_sz], initializer=self.init)
            self.W = tf.get_variable('W', shape=[self.mem_sz, self.mem_sz], initializer=self.init)
    
    def forward(self, inp, msk, initial_state=None, time_major=False, return_final=False, scope='GatedGRU'):
        """ to build the graph, run forward """
        if not time_major:
            inp = tf.transpose(inp, [1, 0, 2])
            msk = tf.transpose(msk, [1, 0])
        
        # after transposition, the shape is [seqlen, batch_size, inp_size]
        batch_size = tf.shape(inp)[1]

        if initial_state is None:
            initial_state = self.zero_state(batch_size, dtype=tf.float32)

        states = tf.scan(self._mask_step,
                elems=[inp, msk],
                initializer=initial_state,
                )
        return states

    def zero_state(self, batch_size, dtype):
        return [tf.tile(tf.expand_dims(key, 0), [batch_size, 1]) for key in self.keys]
    
    @property
    def state_size(self):
        return [self.mem_sz for _ in range(self.m)]

    @property
    def output_size(self):
        return [self.mem_sz for _ in range(self.m)]

    def _mask_step(self, states_tm1, inputs):
        x_t, m_t = inputs
        states = self._step(states_tm1, x_t)
        states = [tf.where(tf.equal(m_t, 1), states[i], states_tm1[i]) for i in range(self.m)]
        return states

    def _step(self, states_tm1, inp_t):
        new_states = []
        for block_id, h in enumerate(states_tm1):
            content_g = tf.reduce_sum(tf.multiply(inp_t, h), axis=[1])
            address_g = tf.reduce_sum(tf.multiply(inp_t, 
                tf.expand_dims(self.keys[block_id], 0)), axis=[1])
            g = tf.sigmoid(content_g + address_g)
            h_component = tf.matmul(h, self.U)
            w_component = tf.matmul(tf.expand_dims(self.keys[block_id], 0), self.V)
            s_component = tf.matmul(inp_t, self.W)
            candidate = self.activation(h_component + w_component + s_component)
            new_h = h + tf.multiply(tf.expand_dims(g, -1), candidate)
            new_h_norm = tf.nn.l2_normalize(new_h, -1)
            new_states.append(new_h_norm)
        return new_states

    def __call__(self, inputs, states):
        """ change order """
        new_states = self._step(states, inputs)
        return new_states, new_states

if __name__ == '__main__':
    keys = [tf.get_variable("Key_%d" % i, [4]) 
                                     for i in range(3)]
    print(keys)
    net = EntityNetwork(3,4, keys=keys)
    inp = tf.zeros([10, 11, 4])
    msk = tf.ones([10, 11])
    y = net.forward(inp, msk, return_final=False)
    print(type(y[0]))

    sess = tf.Session()
    #sess.run(tf.global_variable_initializer())
    sess.run(tf.global_variables_initializer())
    #print(sess.run(y))
    res = sess.run(y)
    import numpy as np
    print(res)
    print(np.shape(res[1]))


    a, b = tf.nn.dynamic_rnn(cell=net,
                        inputs=inp,
                        dtype=tf.float32,
                        #initial_state=cell.zero_state(tf.shape(emb_inp)[0], dtype=tf.float32),
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
    sess.run(tf.global_variables_initializer())
    r0, r1 = sess.run([a, b])
    for x in r0:
        print(x.shape)
    for x in r1:
        print(x.shape)
