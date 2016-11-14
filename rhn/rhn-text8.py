# Attempt at replicating the results from 'Recurrent Highway Networks' using keras
# Arxiv paper: https://arxiv.org/abs/1607.03474
# Reference implementation: https://github.com/julian121266/RecurrentHighwayNetworks
#
import time
import numpy as np

import keras.optimizers
from keras.layers import Embedding, Dense, LSTM, TimeDistributed
from keras.models import Sequential
from rhn import RHN
from lstm_ln import LSTM_LN
from lstm_zoneout import LSTM_zoneout

seq_len = 180
batch_size = 128
epochs = 1000
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

def batchXY(start_idx, length, slen=seq_len):
  Xs = np.zeros((length, dim), dtype='float32')
  Xs[np.arange(length), data[start_idx:start_idx+length]] = 1
  X, Y = [], []
  for idx in xrange(0, length-slen, slen):
    X.append(Xs[idx:idx+slen, :])
    Y.append(Xs[idx+1:idx+slen+1])
  return np.array(X), np.array(Y)

model_name = 'lstm'
train_lbatch = 18
lbatch_size = 5*10**6
validX, validY = batchXY(train_lbatch*lbatch_size, lbatch_size, slen=4096)
print "Valid", np.shape(validX), np.shape(validY)

model = Sequential()
input_shape=(None, dim)
if model_name == 'rhn':
  model.add(RHN(rhn_size, 2, return_sequences=True, consume_less='cpu', input_shape=input_shape))
elif model_name == 'lstm-zoneout':
  model.add(LSTM_zoneout(rhn_size, zoneout_c=0.5, zoneout_h=0.05,
    return_sequences=True, consume_less='gpu', input_shape=input_shape))
elif model_name == 'lstm':
  model.add(LSTM(rhn_size, return_sequences=True, consume_less='gpu', input_shape=input_shape))
else:
  raise(Exception('Unknown model %s' % model_name))

model.add(TimeDistributed(Dense(dim, activation='softmax'), input_shape=(None, rhn_size)))
optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1.0)

print "Compiling model..."
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

bpcs = []
for epoch_idx in xrange(epochs):
  start_time = time.time()
  for cnt in xrange(train_lbatch):
    X, Y = batchXY(cnt * lbatch_size, lbatch_size)
    model.fit(X, Y, batch_size=batch_size, nb_epoch=1)
  # Recompute the loss on the last batch
  train_loss = model.evaluate(X, Y, batch_size=batch_size)
  loss = model.evaluate(validX, validY, batch_size=batch_size)
  bpcs.append((train_loss[0]/np.log(2), loss[0]/np.log(2)))
  print epoch_idx, time.time() - start_time
  print bpcs
  model.save('iter%d.h5' % epoch_idx)
