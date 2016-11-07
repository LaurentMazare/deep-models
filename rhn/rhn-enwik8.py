# Attempt at replicating the results from 'Recurrent Highway Networks' using keras
# Arxiv paper: https://arxiv.org/abs/1607.03474
# Reference implementation: https://github.com/julian121266/RecurrentHighwayNetworks
#
import time
import numpy as np

import keras.optimizers
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from rhn import RHN
from lstm_ln import LSTM_LN

def subsequences(data, seqlen):
  data_shape = np.shape(data)
  shape = data_shape[0] - seqlen + 1, seqlen
  strides = [ data.strides[0] ] + list(data.strides)
  xs = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
  ys = data[seqlen:]
  return xs[:-1], ys

max_length = 50
batch_size = 100
epochs = 10
rhn_size = 256

def load(filename, valid_len=5000000, test_len=5000000):
  with open(filename, 'r') as f:
    data = f.read()
  data = np.fromstring(data, dtype=np.uint8)
  unique, data = np.unique(data, return_inverse=True)
  train_data = data[: -valid_len - test_len]
  valid_data = data[-valid_len - test_len : -test_len]
  test_data = data[-test_len:]
  return train_data, valid_data, test_data, len(unique)

print 'Loading data...'
train_data, valid_data, test_data, dim = load('text8')
print dim
train_x, train_y = subsequences(train_data, max_length)
valid_x, valid_y = subsequences(valid_data, max_length)
test_x, test_y = subsequences(test_data, max_length)
print 'train', np.shape(train_x), np.shape(train_y)
print 'valid', np.shape(train_x), np.shape(train_y)
print 'test', np.shape(test_x), np.shape(test_y)

model = Sequential()
model.add(Embedding(dim, dim, input_length=max_length, dropout=0.2))
if False:
  model.add(RHN(rhn_size, 2, dropout_W=0.2, dropout_U=0.2, consume_less='cpu'))
else:
  model.add(LSTM_LN(rhn_size, dropout_W=0.2, dropout_U=0.2, consume_less='gpu'))

model.add(Dense(dim, activation='softmax'))
optimizer = keras.optimizers.SGD(lr=0.2)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

start_time = time.time()
psize = 25000
losses = []
for idx in xrange(1000000):
  history = model.fit(train_x[idx*psize:(idx+1)*psize],
                      train_y[idx*psize:(idx+1)*psize],
                      batch_size=batch_size,
                      nb_epoch=1)
  losses.append(history.history['loss'][0])
  print idx, losses
#                      validation_data=(valid_x, valid_y))
print (time.time() - start_time) / epochs
