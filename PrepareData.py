import numpy as np
import os
import shutil

def createDir(data_dir):
    if not os.path.isdir(data_dir + '/' + 'images'):
        os.makedirs(data_dir + '/' + 'images')
    if not os.path.isdir(data_dir + '/' + 'labels'):
        os.makedirs(data_dir + '/' + 'labels')

testFile = 'TestSet1.txt'
data_dir = './SiftFlow_Oct30'
output_dir = './SiftFlow_Oct30_new'
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)

train_dir = output_dir + '/' + 'train'

val_dir = output_dir + '/' + 'validation'
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)
if not os.path.isdir(val_dir):
    os.makedirs(val_dir)

createDir(train_dir)
createDir(val_dir)

img_dir = '%s/images'% data_dir
labels_dir = '%s/labels'% data_dir

f = open(testFile)

lines = [line.rstrip('\n') for line in f]
testFilename = [filename.split('\\')[1] for filename in lines]

files = os.listdir(img_dir)

filenames = []
labels = []

count = -1

num_samples = len(files)

perm = np.random.permutation(num_samples-len(testFilename))
remove_idx = perm[0:56]
# Construct the list of JPEG files and labels.
for text in files:

    name, _ = os.path.splitext(os.path.basename(text))
    name = name + '.jpg'
    src_file_path = '%s/%s' % (img_dir, text)
    src_label_path = '%s/%s' % (labels_dir, text)
    if name in testFilename:
        dst_file_path = '%s/%s' % (val_dir + '/images', text)
        dst_label_path = '%s/%s' % (val_dir + '/labels', text)
        shutil.copy(src_file_path, dst_file_path)
        shutil.copy(src_label_path, dst_label_path)
    else:
        count = count + 1
        if not count in remove_idx:
            dst_file_path = '%s/%s' % (train_dir + '/images', text)
            dst_label_path = '%s/%s' % (train_dir + '/labels', text)
            shutil.copy(src_file_path, dst_file_path)
            shutil.copy(src_label_path, dst_label_path)





