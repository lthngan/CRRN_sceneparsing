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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time


import numpy as np
from skimage.color import label2rgb
import tensorflow as tf
import os
import scipy
#from tensorflow.models.image.cifar10 import cifar10

import image_processing
import SiftFlowData
from SiftFlowData import SiftFlowData
from ADEData import ADEData
from CamVidData import CamVidData
from StanfordData import StanfordData

import DAGResnet
import shutil
COLORS = (
'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'gray',
'green', 'greenyellow', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue',
'mediumspringgreen', 'mediumvioletred', 'midnightblue', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal',
'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                          """Either 'validation' or 'train'.""")
tf.app.flags.DEFINE_string('out_label_dir', './results',
                          """Output directory for saving the label results.""")
tf.app.flags.DEFINE_string('prob_dir', './data/prob_npy',
                          """Probability directory.""")

count = -1

def makeDir(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def saveLabelResults(images, labels, predictions, filenames, logits):
  img_path = FLAGS.out_label_dir + '/' + 'images'
  label_path = FLAGS.out_label_dir + '/' + 'labels'
  label_npy = FLAGS.out_label_dir + '/' + 'labels_npy'
  predict_path = FLAGS.out_label_dir + '/' + 'predictions'
  predict_npy = FLAGS.out_label_dir + '/' + 'predictions_npy'
  predict_combine_path = FLAGS.out_label_dir + '/' + 'combined'
  predict_with_DAG_path = FLAGS.out_label_dir + '/' + 'predict_DAG_max'
  predict_with_DAG_path_npy = FLAGS.out_label_dir + '/' + 'predict_DAG_max_npy'
  gt_img_path = FLAGS.out_label_dir + '/' + 'gt_images'

  # val_probs_path = '../../Data/SiftFlow_Oct30_new/validation/val_prob_npy'
  val_probs_path = FLAGS.prob_dir

  makeDir(img_path)
  makeDir(label_path)
  makeDir(predict_path)
  makeDir(predict_combine_path)
  makeDir(label_npy)
  makeDir(predict_npy)
  makeDir(predict_with_DAG_path)
  makeDir(predict_with_DAG_path_npy)
  makeDir(gt_img_path)


  nSamples = images.shape[0]
  global count
  for i in range(nSamples):
    image = images[i]
    image = image * 128 + 128
    image = image.astype(np.uint8)
    image_label_overlay = label2rgb(predictions[i], image=image, colors=COLORS)
    count = count + 1
    scipy.misc.imsave(img_path + '/' + filenames[i], image)
    scipy.misc.imsave(label_path + '/' + filenames[i], labels[i])
    scipy.misc.imsave(predict_path + '/' + filenames[i], predictions[i])
    np.save(label_npy + '/' + filenames[i][0:-3] + 'npy', labels[i])
    np.save(predict_npy + '/' + filenames[i][0:-3] + 'npy', predictions[i])
    scipy.misc.imsave(predict_combine_path + '/' + filenames[i], image_label_overlay)

    scipy.misc.imsave(gt_img_path + '/' + filenames[i], label2rgb(labels[i], image=image, colors=COLORS))

    DAG_probs = np.load(val_probs_path + '/' + filenames[i][0:-3] + 'npy')
    a = logits[i]
    #a[:, :, 1 : a.shape[2]] = np.maximum(a[:, :, 1 : a.shape[2]], DAG_probs)
    a[:, :, 1: a.shape[2]] = np.add(a[:, :, 1: a.shape[2]], DAG_probs)
    labels_new = np.argmax(a, axis=len(a.shape)-1)
    scipy.misc.imsave(predict_with_DAG_path + '/' + filenames[i], label2rgb(labels_new, image=image, colors=COLORS))
    np.save(predict_with_DAG_path_npy + '/' + filenames[i][0:-3] + 'npy', labels_new)


def eval_once(saver, summary_writer, label_predict, summary_op, labels, images, filenames, logits):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        #predictions = sess.run([top_k_op])
        predictions, image, label, filename, logit = sess.run([label_predict, images, labels, filenames, logits])
        #image = sess.run(images)
        #label = sess.run(labels)
        saveLabelResults(image, label, predictions, filename, logit)
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels, filenames = image_processing.distorted_inputs(dataset)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #logits = cifar10.inference(images)
    # Build inference Graph.
    logits = DAGResnet.inference(images)
    logits = tf.nn.softmax(logits)
    shape = logits.get_shape().as_list()
    label_predict = tf.argmax(logits, dimension=len(shape) - 1)
    # Calculate predictions.
    #top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        DAGResnet.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, label_predict, summary_op, labels, images, filenames, logits)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.database == 'SiftFlow':
    dataset = SiftFlowData(subset=FLAGS.subset)
  elif FLAGS.database == 'ADEChallenge':
    dataset = ADEData(subset=FLAGS.subset)
  elif FLAGS.database == 'CamVid':
    dataset = CamVidData(subset=FLAGS.subset)
  elif FLAGS.database == 'Stanford':
    dataset = StanfordData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  if os.path.isdir(FLAGS.out_label_dir):
    shutil.rmtree(FLAGS.out_label_dir)
  evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()
