import tensorflow as tf

class GRU(object):
    def __init__(self, inp_size, num_units):
        """
        GRU implemetation. 
        Args:
            inp_size: input_size
            hidden_size: number of units of hidden variable
        """
        # intializer
        self.inp_size = inp_size
        self.num_units = num_units
        xav_init = tf.contrib.layers.xavier_initializer

        # parameters
        self.W = tf.get_variable('W', shape=[3, num_units, num_units], )#initializer=xav_init())
        self.U = tf.get_variable('U', shape=[3, inp_size, num_units], )#initializer=xav_init())
        self.b = tf.get_variable('b', shape=[3, num_units], initializer=tf.constant_initializer(0.))

    def forward(self, inp, msk, initial_state=None, time_major=False, return_final=False, scope='GatedLSTM'):
        """ to build the graph, run forward """
        if not time_major:
            inp = tf.transpose(inp, [1, 0, 2])
            msk = tf.transpose(msk, [1, 0])
        
        # after transposition, the shape is [seqlen, batch_size, inp_size]
        batch_size = tf.shape(inp)[1]

        if initial_state is None:
            initial_state = self.zero_state(batch_size)

        states = tf.scan(self._step,
                elems=[inp, msk],
                initializer=initial_state,
                )
        
        if return_final:
            states = states[-1]
        else:
            states = tf.transpose(states, [1, 0, 2])

        return states

    def zero_state(self, batch_size):
        return tf.zeros([batch_size, self.num_units])

    def _step(self, prev_s, inputs):
        x_t, m_t = inputs
        z = tf.sigmoid(tf.matmul(x_t, self.U[0]) + tf.matmul(prev_s, self.W[0]) + self.b[0]) 
        r = tf.sigmoid(tf.matmul(x_t, self.U[1]) + tf.matmul(prev_s, self.W[1]) + self.b[1]) 
        h = tf.tanh(tf.matmul(x_t, self.U[2]) + tf.matmul(r * prev_s, self.W[2]) + self.b[2])
        s = (1 - z) * h + z * prev_s
        s = m_t[:, None] * s + (1 - m_t[:, None]) * prev_s
        return s

if __name__ == '__main__':
    net = GRU(3,4)
    inp = tf.zeros([10, 11, 3])
    msk = tf.ones([10, 11])
    y = net.forward(inp, msk, return_final=False)
    print(type(y[0]))

    sess = tf.Session()
    #sess.run(tf.global_variable_initializer())
    sess.run(tf.initialize_all_variables())
    #print(sess.run(y))
    res = sess.run(y)
    import numpy as np
    print(np.shape(res))
