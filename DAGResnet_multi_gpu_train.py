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

"""A binary to train CIFAR-10 using multiple GPU's with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
# import SiftFlow
import image_processing
from SiftFlowData import SiftFlowData
from ADEData import ADEData
from CamVidData import CamVidData
from StanfordData import StanfordData
from FacialHair import FacialHair
import DAGResnet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('do_val', -1,
                            """Do validation after do_val steps. [Default: -1 [no validation]].""")

tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
# add argument --database_name
tf.app.flags.DEFINE_string('database_name', 'SiftFlow',
                           """Database name [Default: SiftFlow].""")

# add argument --data_dir for input data directory
# tf.app.flags.DEFINE_string('data_dir', './data',
#                           """Data directory containing images. [Default: ./data]""")
# add argument --save_dir for output data directory
tf.app.flags.DEFINE_string('save_dir', './save',
                           """Directory to store saved models. [Default: ./save]""")
# add argument --checkpoint_dir for checkpoint directory
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint',
                           """Directory to store checkpointed models. [Default: ./checkpoint]""")

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', './pretrained',
                           """Directory to store pretrained models. [Default: ./pretrained]""")

# add argument --val_step indicating how many steps before to run the validation
tf.app.flags.DEFINE_integer('val_step', 10,
                            """Run validation after val_step steps [Default: 10].""")
# add argument --snapshot_step indicating how many steps before to run the validation
tf.app.flags.DEFINE_integer('snapshot_step', 10,
                            """Take a snapshot after snapshot_step steps [Default: 10].""")
# add argument --testFileList - list of testing filename
tf.app.flags.DEFINE_string('testFileList', 'TestSet1.txt',
                           """List of testing filenames [Default: TestSet1.txt].""")

# add argument --decay_rate
tf.app.flags.DEFINE_float('decay_rate', 0.97,
                          """Decay rate [Default: 0.97].""")
# add argument --decay_rate
# tf.app.flags.DEFINE_integer('batch_size', 64,
#                            """Batch Size [Default: 64].""")
# add argument --num_epochs
tf.app.flags.DEFINE_integer('num_epochs', 200,
                            """Number of epochs [Default: 200].""")

tf.app.flags.DEFINE_integer('start_gpu_idx', 0,
                            """The index of the first gpu will be use. [Default: 0].""")


#def computeAcc(logits, labels):
#    shape = logits.get_shape().as_list()
#    label_predict = tf.argmax(logits, dimension=len(shape) - 1)

def tower_loss(images, labels, scope):
    """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

    # Build inference Graph.
    logits = DAGResnet.inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = DAGResnet.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % DAGResnet.TOWER_NAME, '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(loss_name + ' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(dataset, dataset_val=None):
    """Train CIFAR-10 for a number of steps."""
    #with tf.variable_scope("CRRN", reuse=None):
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (dataset.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * DAGResnet.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(DAGResnet.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        DAGResnet.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        # images, labels = cifar10.distorted_inputs()
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')
        split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

        # Override the number of preprocessing threads to account for the increased
        # number of GPU towers.
        num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
        images, labels, filenames = image_processing.distorted_inputs(
            dataset,
            num_preprocess_threads=num_preprocess_threads)

        # Split the batch of images and labels for towers.
        # images_splits = tf.split(0, FLAGS.num_gpus, images)
        # labels_splits = tf.split(0, FLAGS.num_gpus, labels)
        # Modify because of different version of TF. Date June 15, 2017
        images_splits = tf.split(images, FLAGS.num_gpus, 0)
        labels_splits = tf.split(labels, FLAGS.num_gpus, 0)

        if dataset_val is not None:
            images_val, labels_val, filenames_val = image_processing.distorted_inputs(
                dataset_val, num_preprocess_threads=num_preprocess_threads)
            images_val_splits = tf.split(0, FLAGS.num_gpus, images_val)
            labels_val_splits = tf.split(0, FLAGS.num_gpus, labels_val)

        # Calculate the gradients for each model tower.
        tower_grads = []
        loss_val = []
        pixel_accuracy = []
        for i in xrange(FLAGS.num_gpus):
            gpu_idx = i + FLAGS.start_gpu_idx
            with tf.device('/gpu:%d' % gpu_idx):
                with tf.name_scope('%s_%d' % (DAGResnet.TOWER_NAME, gpu_idx)) as scope:
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    loss = tower_loss(images_splits[i], labels_splits[i], scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)
                    # grads = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads]

                    # grads = [(tf.clip_by_average_norm(grad, 5), var) for grad, var in grads]

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

                    if dataset_val is not None:
                        #with tf.name_scope('Validation'):
                        logits_val = DAGResnet.inference(images_val_splits[i])

                        # Build the portion of the Graph calculating the losses. Note that we will
                        # assemble the total_loss using a custom function below.
                        loss_val.append(DAGResnet.loss(logits_val, labels_val_splits[i]))

                        label_val = labels_val_splits[i]
                        shape = logits_val.get_shape().as_list()
                        label_predict = tf.argmax(logits_val, dimension=len(shape) - 1)
                        pixel_labeled = tf.reduce_sum(tf.to_float(label_val > 0))
                        pixel_correct = tf.reduce_sum(tf.to_float(tf.equal(tf.cast(label_val, tf.int64), label_predict)) * tf.to_float(label_val > 0))
                        pixel_accuracy.append(tf.div(tf.scalar_mul(1.0, pixel_correct), pixel_labeled))




        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        loss_val = tf.reduce_mean(loss_val)
        pixel_accuracy = tf.reduce_mean(pixel_accuracy)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))


        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(
                DAGResnet.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op)
#        train_op = tf.group(apply_gradient_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # restore the previous checkpoint model
        if not tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.MakeDirs(FLAGS.train_dir)
        else:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('%s: Model restored from %s' %
                      (datetime.now(), ckpt.model_checkpoint_path))
            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        # Load the checkpoint model
        if FLAGS.pretrained_model_checkpoint_path:
            try:
                if tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path):
                    t_vars = tf.trainable_variables()
                    variables_to_restore = [var for var in t_vars if not ('FC_V' in var.name or 'upscore' in var.name)]
                    restorer = tf.train.Saver(variables_to_restore)
                    restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
                    print('%s: Pre-trained model restored from %s' %
                          (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
            except ValueError:
                print('No checkpoint is loaded')

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        f = open(FLAGS.train_dir + '/' + 'log.txt', 'w')

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if dataset_val is not None and step > 0 and step % FLAGS.do_val == 0:
                format_str = ('%s: step %d, [VALIDATION] loss = %.6f  pixel acc = %.6f')
                loss_value_val, pixelAcc = sess.run([loss_val, pixel_accuracy])
                print(format_str % (datetime.now(), step, loss_value_val, pixelAcc))
                f.write(format_str % (datetime.now(), step, loss_value_val, pixelAcc))
                f.write('\n')


            if step % 100000 == 0 and step > 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        f.close()


def main(argv=None):  # pylint: disable=unused-argument
    # cifar10.maybe_download_and_extract()
    dataset_val = None
    if FLAGS.database == 'SiftFlow':
        dataset = SiftFlowData(subset=FLAGS.subset)
        if FLAGS.do_val > 0:
            dataset_val = SiftFlowData(subset='validation')
    elif FLAGS.database == 'ADEChallenge':
        dataset = ADEData(subset=FLAGS.subset)
        if FLAGS.do_val > 0:
            dataset_val = ADEData(subset='validation')
    elif FLAGS.database == 'CamVid':
        dataset = CamVidData(subset=FLAGS.subset)
        if FLAGS.do_val > 0:
            dataset = CamVidData(subset='validation')
    elif FLAGS.database == 'Stanford':
        dataset = StanfordData(subset=FLAGS.subset)
        if FLAGS.do_val > 0:
            dataset = StanfordData(subset='validation')
    elif FLAGS.database == 'FacialHair':
        dataset = FacialHair(subset=FLAGS.subset)
        if FLAGS.do_val > 0:
            dataset = FacialHair(subset='validation')
    else:
        print('Database is not supported')
        return
    print(FLAGS.database)

    #if tf.gfile.Exists(FLAGS.train_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #if not tf.gfile.Exists(FLAGS.train_dir):
    #    tf.gfile.MakeDirs(FLAGS.train_dir)
    train(dataset, dataset_val)


if __name__ == '__main__':
    tf.app.run()
