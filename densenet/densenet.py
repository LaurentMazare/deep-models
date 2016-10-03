import numpy as np
import tensorflow as tf

def unpickle(file):
  import cPickle
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  if 'data' in dict:
    dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3) / 256.

  return dict

def load_data_one(f):
  batch = unpickle(f)
  data = batch['data']
  labels = batch['labels']
  print "Loading %s: %d" % (f, len(data))
  return data, labels

def load_data(files, data_dir, label_count):
  data, labels = load_data_one(data_dir + '/' + files[0])
  for f in files[1:]:
    data_n, labels_n = load_data_one(data_dir + '/' + f)
    data = np.append(data, data_n, axis=0)
    labels = np.append(labels, labels_n, axis=0)
  labels = np.array([ [ float(i == label) for i in xrange(label_count) ] for label in labels ])
  return data, labels

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
  conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  if with_bias:
    return conv + bias_variable([ out_features ])
  return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
  current = tf.contrib.layers.batch_norm(current, is_training=is_training, updates_collections=None)
  current = tf.nn.relu(current)
  current = conv2d(current, in_features, out_features, kernel_size)
  current = tf.nn.dropout(current, keep_prob)
  return current

def block(input, layers, in_features, growth, is_training, keep_prob):
  current = input
  features = in_features
  for idx in xrange(layers):
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
    current = tf.concat(3, (current, tmp))
    features += growth
  return current, features

def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def run_model(data, image_dim, label_count, depth):
  weight_decay = 1e-4
  layers = (depth - 4) / 3
  graph = tf.Graph()
  with graph.as_default():
    xs = tf.placeholder("float", shape=[None, image_dim])
    ys = tf.placeholder("float", shape=[None, label_count])
    lr = tf.placeholder("float", shape=[])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder("bool", shape=[])


    current = tf.reshape(xs, [ -1, 32, 32, 3 ])
    current = conv2d(current, 3, 16, 3)

    current, features = block(current, layers, 16, 12, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)
    current, features = block(current, layers, features, 12, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)
    current, features = block(current, layers, features, 12, is_training, keep_prob)

    current = tf.contrib.layers.batch_norm(current, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = avg_pool(current, 8)
    final_dim = features
    current = tf.reshape(current, [ -1, final_dim ])
    Wfc = weight_variable([ final_dim, label_count ])
    bfc = bias_variable([ label_count ])
    ys_ = tf.nn.softmax( tf.matmul(current, Wfc) + bfc )

    cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
    correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  with tf.Session(graph=graph) as session:
    start_batch = 0
    batch_size = 64
    train_data, train_labels = data['train_data'], data['train_labels']
    batch_count = len(train_data) / batch_size
    batches_data = np.split(train_data[:batch_count*batch_size], batch_count)
    batches_labels = np.split(train_labels[:batch_count*batch_size], batch_count)
    learning_rate = 0.1
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    for epoch in xrange(1, 1+300):
      if epoch == 150: learning_rate = 0.01
      if epoch == 225: learning_rate = 0.001
      for batch_idx in xrange(batch_count):
        batch_data = batches_data[batch_idx]
        batch_labels = batches_labels[batch_idx]
      
        batch_res = session.run([ train_step, cross_entropy, accuracy ],
          feed_dict = { xs: batch_data, ys: batch_labels, lr: learning_rate, is_training: True, keep_prob: 0.8 })

      save_path = saver.save(session, 'densenet_%d.ckpt' % epoch)
      total_acc, total_ce = 0, 0
      for i in xrange(50):
        ce, acc = session.run([ cross_entropy, accuracy ],
            feed_dict = { xs: data['test_data'][i*200:(i+1)*200], ys: data['test_labels'][i*200:(i+1)*200], is_training: True, keep_prob: 1. })
        total_acc, total_ce = total_acc + acc, total_ce + ce
      print epoch, batch_res[1:], total_acc / 50, total_ce / 50

def run():
  data_dir = 'data'
  image_size = 32
  image_dim = image_size * image_size * 3
  meta = unpickle(data_dir + '/batches.meta')
  label_names = meta['label_names']
  label_count = len(label_names)

  train_files = [ 'data_batch_%d' % d for d in xrange(1, 6) ]
  train_data, train_labels = load_data(train_files, data_dir, label_count)
  pi = np.random.permutation(len(train_data))
  train_data, train_labels = train_data[pi], train_labels[pi]
  test_data, test_labels = load_data([ 'test_batch' ], data_dir, label_count)
  print "Train:", np.shape(train_data), np.shape(train_labels)
  print "Test:", np.shape(test_data), np.shape(test_labels)
  data = { 'train_data': train_data,
      'train_labels': train_labels,
      'test_data': test_data,
      'test_labels': test_labels }
  run_model(data, image_dim, label_count, 40)

run()
