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

seq_len = 180
batch_size = 128
epochs = 10
rhn_size = 256

def load(filename):
  with open(filename, 'r') as f:
    data = f.read()
  data = np.fromstring(data, dtype=np.uint8)
  unique, data = np.unique(data, return_inverse=True)
  return data, len(unique)

print 'Loading data...'
data, dim = load('text8')
print 'Alphabet size', dim

def batchXY(start_idx, length):
  Xs = np.zeros((length, dim), dtype='float32')
  Xs[np.arange(length), data[start_idx:start_idx+length]] = 1
  X, Y = [], []
  for idx in xrange(0, length-seq_len, seq_len):
    X.append(Xs[idx:idx+seq_len, :])
    Y.append(data[start_idx+idx+seq_len])
  return np.array(X), np.array(Y)

lbatch_size = 5*10**6
validX, validY = batchXY(90*10**6, lbatch_size)
print "Valid", np.shape(validX), np.shape(validY)

model = Sequential()
input_shape=(seq_len, dim)
if False:
  model.add(RHN(rhn_size, 2, dropout_W=0.5, dropout_U=0.5, consume_less='cpu', input_shape=input_shape))
else:
  model.add(LSTM(rhn_size, dropout_W=0.5, dropout_U=0.5, consume_less='gpu', input_shape=input_shape))

model.add(Dense(dim, activation='softmax'))
optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1.)

print "Compiling model..."
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

losses = []
for epoch_idx in xrange(epochs):
  start_time = time.time()
  for cnt in xrange(18):
    X, Y = batchXY(cnt * lbatch_size, lbatch_size)
    model.fit(X, Y, batch_size=batch_size, nb_epoch=1)
  loss = model.evaluate(X, Y, batch_size=batch_size)
  losses.append(loss)
  print epoch_idx, time.time() - start_time
  print losses
