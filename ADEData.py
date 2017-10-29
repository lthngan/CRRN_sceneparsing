# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Small library that points to the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dataset import Dataset
import numpy as np

#WEIGHTS = np.array([0.5, 4, 4, 16, 8, 4, 1, 8, 2, 16, 4, 4, 4, 4, 2, 2, 16, 1, 4, 2, 8, 2, 1, 2, 2, 1, 2, 4, 1, 4, 8, 8, 1, 2])
IMAGE_SIZE = 256

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 151
WEIGHTS = np.ones((NUM_CLASSES,))
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2432
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

class ADEData(Dataset):
  """Flowers data set."""

  def __init__(self, subset):
    super(ADEData, self).__init__('ADEChallenge', subset)
    # Process images of this size. Note that this differs from the original CIFAR
    # image size of 32 x 32. If one alters this number, then the entire model
    # architecture will change and any model would need to be retrained.
    self.IMAGE_SIZE = IMAGE_SIZE

    # Global constants describing the CIFAR-10 data set.
    self.NUM_CLASSES = NUM_CLASSES
    self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return self.NUM_CLASSES

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    if self.subset == 'validation':
      return self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""

    #print('Failed to find any Flowers %s files'% self.subset)
    #print('')
    #print('If you have already downloaded and processed the data, then make '
    #      'sure to set --data_dir to point to the directory containing the '
    #      'location of the sharded TFRecords.\n')
    #print('Please see README.md for instructions on how to build '
    #'the flowers dataset using download_and_preprocess_flowers.\n')