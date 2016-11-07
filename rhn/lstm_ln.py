from keras import backend as K
from keras.layers import LSTM, time_distributed_dense
from keras import initializations, activations, regularizers
from keras.engine import InputSpec

class LSTM_LN(LSTM):
  def __init__(self, output_dim, **kwargs):
    super(LSTM_LN, self).__init__(output_dim, **kwargs)

  def norm(self, xs, norm_id):
    mu = K.mean(xs, axis=-1, keepdims=True)
    sigma = K.sqrt(K.var(xs, axis=-1, keepdims=True) + 1e-7)
    xs = self.gs[norm_id] * (xs - mu) / sigma + self.bs[norm_id]
    return xs

  def build(self, input_shape):
    super(LSTM_LN, self).build(input_shape)
    self.gs, self.bs = [], []
    for i in xrange(5):
      self.gs += [ self.init((self.output_dim,), name='{}_g%i'.format(self.name, i)) ]
      self.bs += [ K.zeros((self.output_dim,), name='{}_b%d'.format(self.name, i)) ]
    self.trainable_weights += self.gs + self.bs

  def step(self, x, states):
    h_tm1 = states[0]
    c_tm1 = states[1]
    B_U = states[2]
    B_W = states[3]

    if self.consume_less == 'gpu':
      z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b

      z0 = z[:, :self.output_dim]
      z1 = z[:, self.output_dim: 2 * self.output_dim]
      z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
      z3 = z[:, 3 * self.output_dim:]

      i = self.inner_activation(self.norm(z0, 0))
      f = self.inner_activation(self.norm(z1, 1))
      c = f * c_tm1 + i * self.activation(self.norm(z2, 2))
      o = self.inner_activation(self.norm(z3, 3))
    else:
      if self.consume_less == 'cpu':
        x_i = x[:, :self.output_dim]
        x_f = x[:, self.output_dim: 2 * self.output_dim]
        x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
        x_o = x[:, 3 * self.output_dim:]
      elif self.consume_less == 'mem':
        x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
        x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
        x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
        x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
      else:
        raise Exception('Unknown `consume_less` mode.')

      i = self.inner_activation(self.norm(x_i + K.dot(h_tm1 * B_U[0], self.U_i), 0))
      f = self.inner_activation(self.norm(x_f + K.dot(h_tm1 * B_U[1], self.U_f), 1))
      c = f * c_tm1 + i * self.activation(self.norm(x_c + K.dot(h_tm1 * B_U[2], self.U_c), 2))
      o = self.inner_activation(self.norm(x_o + K.dot(h_tm1 * B_U[3], self.U_o), 3))

    h = o * self.activation(self.norm(c, 4))
    return h, [h, c]
