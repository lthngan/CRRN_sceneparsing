# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np
from utils import *
from tflearn.layers.normalization import batch_normalization
import SiftFlowData
import ADEData
import CamVidData
import StanfordData
import FacialHair

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
#tf.app.flags.DEFINE_integer('batch_size', 128,
#                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('debug', True,
                            """Using debug mode or not. [Default: True]""")

tf.app.flags.DEFINE_integer('reduce_size', 64,
                            """Downsampling size. [Default: 64]""")

tf.app.flags.DEFINE_boolean('use_attention_weight', False,
                            """Using attention weight or not. [Default: False]""")
tf.app.flags.DEFINE_integer('block_size', 8,
                            """Block size. [Defaule: 8]""")
tf.app.flags.DEFINE_integer('hid_size', 8,
                            """Hidden size. [Defaule: 8]""")
tf.app.flags.DEFINE_integer('direction', -1,
                            """Which direction to go. [-1 for all] [Default: 0]""")
tf.app.flags.DEFINE_boolean('usePredictionMax', False,
                            """Use the maximum value of 4 direction or not. [Default: False]""")
tf.app.flags.DEFINE_string('database', 'SiftFlow',
                            """Select the Database to train [SiftFlow/ ADEChallenge]. [Default: SiftFlow].""")
# add argument --learning_rate
tf.app.flags.DEFINE_float('learning_rate', 0.0002,
                          """Learning rate [Default: 0.0002].""")

# Global constants describing the CIFAR-10 data set.
#IMAGE_SIZE = FLAGS.image_size
#NUM_CLASSES = cifar10_input.NUM_CLASSES
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 500 #350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = FLAGS.learning_rate       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# model parameter

reduced_shape = [FLAGS.reduce_size, FLAGS.reduce_size, 64]
f_downsampling_size = [[8, 8, 3, 32], [5, 5, 32, 64], [5, 5, 64, 64], [4, 4, 64, 64]]
image_shape = [FLAGS.image_size, FLAGS.image_size, 3]
image_shape_orig = [image_shape[0], image_shape[1], image_shape[2]]  # make a copy
block_size=[FLAGS.block_size, FLAGS.block_size]
hidden_size=[FLAGS.hid_size, FLAGS.hid_size, 1]
res_f_size=[3, 3, 1, 1]

if FLAGS.direction == -1:
    prefix = ['SE', 'SW', 'NW', 'NE']
    step_i = [ 1,  1, -1, -1]
    step_j = [ 1, -1, -1,  1]
elif FLAGS.direction == 0:
    prefix = ['SE']
    step_i = [1]
    step_j = [1]
elif FLAGS.direction == 1:
    prefix = ['SW']
    step_i = [1]
    step_j = [-1]
elif FLAGS.direction == 2:
    prefix = ['NW']
    step_i = [-1]
    step_j = [-1]
elif FLAGS.direction == 3:
    prefix = ['NE']
    step_i = [-1]
    step_j = [1]

debug = FLAGS.debug
if FLAGS.database == 'SiftFlow':
    atention_weight = SiftFlowData.WEIGHTS
    num_class = SiftFlowData.NUM_CLASSES
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = SiftFlowData.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
elif FLAGS.database == 'ADEChallenge':
    atention_weight = ADEData.WEIGHTS
    num_class = ADEData.NUM_CLASSES
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ADEData.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
elif FLAGS.database == 'CamVid':
    atention_weight = CamVidData.WEIGHTS
    num_class = CamVidData.NUM_CLASSES
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ADEData.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
elif FLAGS.database == 'Stanford':
    atention_weight = StanfordData.WEIGHTS
    num_class = StanfordData.NUM_CLASSES
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ADEData.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
elif FLAGS.database == 'FacialHair':
    atention_weight = FacialHair.WEIGHTS
    num_class = FacialHair.NUM_CLASSES
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FacialHair.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

else:
    print('Database is not supported! Errors may come out!')


#prefix = ['SE']
#step_i = [ 1]
#step_j = [ 1]

num_block2 = [int(image_shape[0]/ block_size[0]), int(image_shape[1]/ block_size[1])]
num_block = np.prod(num_block2)


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  # tf.histogram_summary(tensor_name + '/activations', x)
  # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
  # Modify because of different version of TF. Date June 15, 2017
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    # weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    # Modify because of different version of TF. Date June 15, 2017
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def add_block_conv(x, name, f_size, stride=[1, 1, 1, 1]):
    """Add a conv layer with batch norm and relu"""
    with tf.variable_scope(name):
        kernel = _variable_with_weight_decay('weights',
                                             shape=f_size,
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(x, kernel, stride, padding='SAME', use_cudnn_on_gpu=True)
        #y = batch_normalization(incoming=conv)
        biases = _variable_on_cpu('biases', [f_size[3]], tf.constant_initializer(0.0))
        y = tf.nn.bias_add(conv, biases)
        # y = relu(y)
        y = batch_normalization(incoming=y)

        return y

def add_block_res(x, prefix, f_size, stride=[1, 1, 1, 1]):

    with tf.variable_scope(prefix, reuse=False):
        y = add_block_conv(x, "conv1", f_size, stride)
        y = relu(y)
        y1 = add_block_conv(y, "conv2", f_size, stride)
        return relu(tf.add(y1, x))

def relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    # return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    # Modify because of different version of TF, Data June 15, 2017
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """Fully connected"""
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = _variable_on_cpu('Weight_Matrix', [shape[1], output_size], tf.random_normal_initializer(stddev=stddev))
        bias = _variable_on_cpu('bias_Matrix', [output_size], tf.constant_initializer(bias_start))
        #matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
        #                         tf.random_normal_initializer(stddev=stddev))
        #bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def add_rnn_unit(prefix, in_data, in_hidden, is_last, unitID=None, isFirst=None):
    """Add a RNN-like unit"""
    # hidden_size: numpy array
    # in_data, in_hidden are lists
    if isFirst is None or isFirst==True:
        flagReuse = False
    else:
        flagReuse = True
#        tf.get_variable_scope().reuse_variables()

    f_size = res_f_size
    n_class = num_class
    num_samples = int(in_data.get_shape()[0])
    with tf.variable_scope(prefix) as scope_prefix:
        n_fc = np.prod(hidden_size)

        # Fully connect for U*x
        with tf.variable_scope("FC_U", reuse=flagReuse) as scope:
            U_mult_x = linear(tf.reshape(in_data, [num_samples, -1]), n_fc)
            U_mult_x = batch_normalization(incoming=U_mult_x)

        # Sum up predecessors of hidden
        #if len(in_hidden) == 1:
        #    h_sum = in_hidden[0]
        #else:
        #h_sum = tf.add_n(in_hidden)
        h_sum = in_hidden

        # Fully connect for W*h
        with tf.variable_scope("FC_W", reuse=flagReuse) as scope:
            W_mult_h = linear(tf.reshape(h_sum, [num_samples, -1]), n_fc)
            W_mult_h = batch_normalization(incoming=W_mult_h)

        with tf.variable_scope("FC_bias", reuse=flagReuse):
            bias = _variable_on_cpu('bias_FC_hid', W_mult_h.get_shape(), tf.constant_initializer(0.0))

        h = relu(tf.add(W_mult_h, U_mult_x) + bias)

        if not is_last:
            # ResNet block
            shape = [num_samples] + hidden_size
            if unitID is not None:
                print('Add block_resnet %s for unit %d' % (prefix, unitID))
            else:
                print('Add block_resnet %s' % prefix)

            if unitID is not None:
                h_res = add_block_res(tf.reshape(h, shape), 'Resnet_unit%d' % unitID, f_size)
            else:
                h_res = add_block_res(tf.reshape(h, shape), scope_prefix, f_size)
            h_res = tf.reshape(h_res, [num_samples, -1])
        else:
            h_res = h

        # Fully connect for V*h
        with tf.variable_scope("FC_V", reuse=flagReuse) as scope:
            shape = in_data.get_shape().as_list()
            n_pixel = shape[1] * shape[2]
            n_volume = n_pixel * n_class
            o = linear(h_res, n_volume)
            o = tf.reshape(o, [num_samples, shape[1], shape[2], n_class])


        return h_res, o

def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """

  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  # downs sampling the images
  images = add_downsampling_conv_layers(images, reduced_shape[0])
  global image_shape
  global prefix

  image_shape = reduced_shape

  # recompute num_block using the reduced shape
  num_block2 = [int(image_shape[0] / block_size[0]), int(image_shape[1] / block_size[1])]
  num_block = np.prod(num_block2)

  # divide the image into blocks
  count = -1
  blocks = [None] * num_block
  num_samples = images.get_shape()[0]._value
  for i in xrange(num_block2[0]):
      for j in xrange(num_block2[1]):
          count += 1
          #blocks[count] = tf.slice(images, [0, i * block_size[0], j * block_size[1], 0], [num_samples] + block_size + [images.get_shape()[3]._value])
          blocks[count] = images[:,i * block_size[0] : (i+1) * block_size[0], j * block_size[1] : (j+1) * block_size[1],:]



  #h0 = _variable_on_cpu('h0', [num_samples] + image_shape[0:2], tf.random_normal_initializer(stddev=0.35))

  #h0 = _variable_on_cpu('h0', [num_samples] + [np.prod(hidden_size)], tf.random_normal_initializer(stddev=0.35))
  h0 = _variable_on_cpu('h0', [num_samples] + [np.prod(hidden_size)], tf.constant_initializer(0.0))


  direction_predict = []
  prediction_max = None
  isFirst = True
  for p in xrange(len(prefix)):
      in_hidden = tf.identity(h0)
      range_i, start_i, stop_i = range_(num_block2[0], step_i[p])
      range_j, start_j, stop_j = range_(num_block2[1], step_j[p])
      hiddens = [[[] for _ in xrange(num_block2[1])] for _ in xrange(num_block2[0])]
      predict = [[[] for _ in xrange(num_block2[1])] for _ in xrange(num_block2[0])]

      direction_predict_lst = []
      with tf.variable_scope(prefix[p]) as scope:

          for i in range_i:
              for j in range_j:
                  k = sub2ind(num_block2, i, j)
                  if i != start_i:
                      if in_hidden is None:
                          in_hidden = hiddens[i - step_i[p]][j]
                      else:
                        in_hidden = tf.add(in_hidden, hiddens[i - step_i[p]][j])
                  if j != start_j:
                      if in_hidden is None:
                          in_hidden = hiddens[i][j - step_j[p]]
                      else:
                        in_hidden = tf.add(in_hidden, hiddens[i][j - step_j[p]])
                  if i != start_i and j != start_j:
                      in_hidden = tf.add(in_hidden, hiddens[i - step_i[p]][j - step_j[p]])
                  is_last = (i == stop_i) and (j == stop_j)
                  # add_rnn_unit
                  hiddens[i][j], predict[i][j] = add_rnn_unit("unit", blocks[k], in_hidden, is_last, k, isFirst)
                  isFirst = False

                  in_hidden = None#_variable_on_cpu('pre_hid%d' % k, [num_samples] + image_shape[0:2],
                  #                             tf.constant_initializer(0.0))
              # layer_out = tf.concat(2, predict[i])
              # Modify because of different version of TF Date June 15, 2017
              layer_out = tf.concat(predict[i], 2)
              direction_predict_lst.append(layer_out)
      isFirst = True
      # predict_logits = tf.concat(1, direction_predict_lst)
      # Modify because of different version of TF Date June 15, 2017
      predict_logits = tf.concat(direction_predict_lst, 1)
      layer_out = tf.concat(predict[i], 2)
      direction_predict.append(predict_logits)
      if prediction_max is None:
          prediction_max = tf.identity(predict_logits)
      else:
          prediction_max = tf.maximum(prediction_max, predict_logits)

  if not FLAGS.usePredictionMax:
    final_outputs = tf.scalar_mul(1.0/len(prefix), tf.add_n(direction_predict))
  else:
    final_outputs = prediction_max

  upsample = image_shape_orig[1] // reduced_shape[1]
  final_outputs = _upscore_layer(final_outputs,
                                 shape=[num_samples] + image_shape_orig[0:2] + [num_class],
                                 num_classes=num_class,
                                 debug=debug, name='upscore',
                                 ksize=upsample * 2, stride=upsample)

  return final_outputs



def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  if FLAGS.use_attention_weight:
    #labels = tf.one_hot(labels, num_class)
    #labels = tf.mul(labels, atention_weight)
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #       logits, labels, name='cross_entropy_per_example')
    logits = tf.multiply(logits, atention_weight)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #    logits, labels, name='cross_entropy_per_example')
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
  else:
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
     #   logits, labels, name='cross_entropy_per_example')
    cross_entropy =  tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = labels, name='cross_entropy_per_example')
  #cross_entropy = tf.contrib.losses.sparse_softmax_cross_entropy(
  #    logits, labels, weight=atention_weight, scope='cross_entropy_per_example')
  #cross_entropy = tf.contrib.compute_weighted_loss(cross_entropy, weight=atention_weight)
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return _variable_on_cpu('up_filter', weights.shape, initializer=init)
    #return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)


def _add_deconv_layer(bottom, name, f_shape, output_shape):
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, strides=strides, padding='SAME')
    return deconv


def add_downsampling_conv_layers(x, output_shape):
    f_size = f_downsampling_size
    shape = x.get_shape().as_list()[1]
    y = x
    cnt = 0
    while shape > output_shape:
        y = add_block_conv(y, "conv%d" % cnt, f_size[cnt])
        # y = max_pool_2d(y, 2)
        y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        cnt += 1
        shape //= 2

    return y


def _upscore_layer(bottom, shape, num_classes, name, debug, ksize=4, stride=2, wd=5e-4):

    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        #h = ((shape[1] - 1) * stride) + 1
        #w = ((shape[2] - 1) * stride) + 1
        #new_shape = [shape[0], h, w, num_classes]

        if shape is None:
            # Compute shape out of Bottom
            # in_shape = tf.shape(bottom)
            in_shape = bottom.get_shape().as_list()

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]

        output_shape = new_shape
        # output_shape = tf.pack(new_shape)

        #logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input) ** 0.5

        weights = get_deconv_filter(f_shape)
        #_add_wd_and_summary(weights, wd, "fc_wlosses")
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, strides=strides, padding='SAME')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)], message='Shape of %s' % name, summarize=4, first_n=1)

    _activation_summary(deconv)
    return deconv


def _add_wd_and_summary(var, wd, collection_name="losses"):
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    _variable_summaries(var)
    return var


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)
