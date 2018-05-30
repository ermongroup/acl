import tensorflow as tf

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter
  
def weight_variable(name, shape, stddev=0.03):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape=shape, initializer=initializer)

def bias_variable(name, shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable(name, shape=shape, initializer=initializer)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
                          
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def linear(input_, output_size, keep_prob = 1., scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    input_ = tf.nn.dropout(input_, keep_prob=keep_prob)
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
                                                
def conv2d_new(input_, output_dim, 
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="conv2d_new"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv
    
def deconv2d(input_, output_shape,
       k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
          
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)
    
def make_feat(x_image, im_height, keep_prob):
    #x2 = weight_variable("xx", [16*5, 56, 56, 3])
    W_conv1 = weight_variable("W_conv1", [5, 5, 3, 32])
    b_conv1 = bias_variable("b_conv1", [32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable("W_conv2", [5, 5, 32, 64])
    b_conv2 = bias_variable("b_conv2", [64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    if im_height == 112:
        W_conv3 = weight_variable("W_conv3", [5, 5, 64, 64])
        b_conv3 = bias_variable("b_conv3", [64])

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        h_pool_final = h_pool3
        scale = im_height / 56
    elif im_height == 56:
        h_pool_final = h_pool2
        scale = im_height / 28
    elif im_height == 64:
        h_pool_final = h_pool2
        scale = im_height / 32
    else:
        assert False

    W_fc1 = weight_variable("W_fc1", [8 * 8 * 64 * scale ** 2, 256])#7*7
    b_fc1 = bias_variable("b_fc1", [256])

    h_pool_final_flat = tf.reshape(h_pool_final, [-1, 8*8*64 * scale ** 2])#7*7
    h_pool_final_drop = tf.nn.dropout(h_pool_final_flat, keep_prob=keep_prob)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_final_drop, W_fc1) + b_fc1)

    return h_fc1


def build_lstm_inner(lstm_input, batch_size, state_size, rnn_len):
    '''
    build lstm decoder
    '''
    lstm = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=0.0, state_is_tuple=False)

    state = tf.zeros([batch_size, lstm.state_size])

    outputs = []
    with tf.variable_scope('RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        for time_step in range(rnn_len):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input[:, time_step, :], state)
            outputs.append(output)
    return outputs
