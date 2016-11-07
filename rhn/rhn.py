from keras import backend as K
from keras.layers import Recurrent, time_distributed_dense
from keras import initializations, activations, regularizers
from keras.engine import InputSpec

# This is a copied and modified version of keras LSTM.
# only the 'cpu' and 'mem' consume_less target are supported.
class RHN(Recurrent):
  def __init__(self, output_dim, L,
             init='glorot_uniform', inner_init='orthogonal',
             activation='tanh', inner_activation='hard_sigmoid',
             W_regularizer=None, U_regularizer=None, b_regularizer=None,
             dropout_W=0., dropout_U=0., **kwargs):
    self.output_dim = output_dim
    self.init = initializations.get(init)
    self.inner_init = initializations.get(inner_init)
    self.activation = activations.get(activation)
    self.inner_activation = activations.get(inner_activation)
    self.W_regularizer = regularizers.get(W_regularizer)
    self.U_regularizer = regularizers.get(U_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)
    self.dropout_W, self.dropout_U = dropout_W, dropout_U
    self.L = L

    if self.dropout_W or self.dropout_U:
        self.uses_learning_phase = True
    super(RHN, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_spec = [InputSpec(shape=input_shape)]
    self.input_dim = input_shape[2]

    if self.stateful:
      self.reset_states()
    else:
      # initial states: all-zero tensor of shape (output_dim)
      self.states = [None]

    self.W_t = self.init((self.input_dim, self.output_dim),
                         name='{}_W_t'.format(self.name))
    self.b_t = K.zeros((self.output_dim,), name='{}_b_t'.format(self.name))
    self.W_h = self.init((self.input_dim, self.output_dim),
                         name='{}_W_h'.format(self.name))
    self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

    self.U_ts, self.b_ts = [], []
    self.U_hs, self.b_hs = [], []
    for l in xrange(self.L):
      self.U_ts.append(self.inner_init((self.output_dim, self.output_dim), name='{}_U_t{}'.format(self.name, l)))
      self.b_ts.append(K.zeros((self.output_dim,), name='{}_b_t{}'.format(self.name, l)))
      self.U_hs.append(self.inner_init((self.output_dim, self.output_dim), name='{}_U_h{}'.format(self.name, l)))
      self.b_hs.append(K.zeros((self.output_dim,), name='{}_b_h{}'.format(self.name, l)))

    self.trainable_weights = [ self.W_t, self.b_t, self.W_h, self.b_h] + self.U_ts + self.U_hs + self.b_ts + self.b_hs

    self.W = K.concatenate([self.W_t, self.W_h])
    self.U = K.concatenate(self.U_ts + self.U_hs)
    self.b = K.concatenate([self.b_t, self.b_h] + self.b_ts + self.b_hs)

    self.regularizers = []
    if self.W_regularizer:
      self.W_regularizer.set_param(self.W)
      self.regularizers.append(self.W_regularizer)
    if self.U_regularizer:
      self.U_regularizer.set_param(self.U)
      self.regularizers.append(self.U_regularizer)
    if self.b_regularizer:
      self.b_regularizer.set_param(self.b)
      self.regularizers.append(self.b_regularizer)

    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights

  def reset_states(self):
    assert self.stateful, 'Layer must be stateful.'
    input_shape = self.input_spec[0].shape
    if not input_shape[0]:
      raise Exception('If a RNN is stateful, a complete ' +
                      'input_shape must be provided (including batch size).')
    if hasattr(self, 'states'):
      K.set_value(self.states[0],
                  np.zeros((input_shape[0], self.output_dim)))
    else:
      self.states = [K.zeros((input_shape[0], self.output_dim))]

  def preprocess_input(self, x):
    if self.consume_less == 'cpu':
      input_shape = self.input_spec[0].shape
      input_dim = input_shape[2]
      timesteps = input_shape[1]

      x_t = time_distributed_dense(x, self.W_t, self.b_t, self.dropout_W,
                                   input_dim, self.output_dim, timesteps)
      x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
                                   input_dim, self.output_dim, timesteps)
      return K.concatenate([x_t, x_h], axis=2)
    else:
      return x

  def step(self, x, states):
    h_st, B_U, B_W = states

    if self.consume_less == 'cpu':
      x_t = x[:, :self.output_dim]
      x_h = x[:, self.output_dim: 2 * self.output_dim]
    elif self.consume_less == 'mem':
      x_t = K.dot(x * B_W[0], self.W_t) + self.b_t
      x_h = K.dot(x * B_W[1], self.W_h) + self.b_h
    else:
      raise Exception('Unknown `consume_less` mode.')

    for l in xrange(self.L):
      if l == 0:
        t = self.inner_activation(x_t + K.dot(h_st * B_U[0], self.U_ts[l]) + self.b_ts[l])
        h = self.activation(x_h + K.dot(h_st * B_U[1], self.U_hs[l]) + self.b_hs[l])
      else:
        t = self.inner_activation(K.dot(h_st * B_U[0], self.U_ts[l]) + self.b_ts[l])
        h = self.activation(K.dot(h_st * B_U[1], self.U_hs[l]) + self.b_hs[l])
      h_st = h * t + h_st * (1 - t)

    return h_st, [h_st]

  def get_constants(self, x):
    constants = []
    if 0 < self.dropout_U < 1:
      ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, self.output_dim))
      B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
      constants.append(B_U)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(3)])

    if 0 < self.dropout_W < 1:
      input_shape = self.input_spec[0].shape
      input_dim = input_shape[-1]
      ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, input_dim))
      B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
      constants.append(B_W)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(3)])
    return constants

  def get_config(self):
    config = {'output_dim': self.output_dim,
              'init': self.init.__name__,
              'inner_init': self.inner_init.__name__,
              'activation': self.activation.__name__,
              'inner_activation': self.inner_activation.__name__,
              'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
              'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
              'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
              'dropout_W': self.dropout_W,
              'dropout_U': self.dropout_U}
    base_config = super(RHN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


