# LSTM implementation using zoneout as described in
# Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations
# https://arxiv.org/abs/1606.01305
from keras import backend as K
from keras.layers import LSTM, time_distributed_dense
from keras import initializations, activations, regularizers
from keras.engine import InputSpec

class LSTM_zoneout(LSTM):
  def __init__(self, output_dim, zoneout_h=0., zoneout_c=0., **kwargs):
    self.zoneout_h = zoneout_h
    self.zoneout_c = zoneout_c
    if self.zoneout_h or self.zoneout_c:
      self.uses_learning_phase = True
    super(LSTM_zoneout, self).__init__(output_dim, **kwargs)

  def zoneout(self, v, prev_v, pr=0.):
    diff = v - prev_v
    diff = K.in_train_phase(K.dropout(diff, pr, noise_shape=(self.output_dim,)), diff)
    # In testing, always return v * (1-pr) + prev_v * pr
    # In training when K.dropout returns 0, return prev_v
    #             when K.dropout returns diff/(1-pr), return v
    return prev_v + diff * (1-pr)

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

      i = self.inner_activation(z0)
      f = self.inner_activation(z1)
      c = f * c_tm1 + i * self.activation(z2)
      o = self.inner_activation(z3)
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

      i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
      f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
      c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
      o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

    if self.zoneout_c:
      c = self.zoneout(c, c_tm1, pr=self.zoneout_c)
    h = o * self.activation(c)
    if self.zoneout_h:
      h = self.zoneout(h, h_tm1, pr=self.zoneout_h)
    return h, [h, c]
