import os
import deepdish as dd
import numpy as np
import scipy.io as sio
from scipy import misc
import scipy
import cv2


def create_imdb_siftflowEx(data_dir, label_dir, imdb_path='imdb.h5', is_save=True):
    """Create image dataset"""

    list_data = os.listdir(data_dir)
    # list_label = os.listdir(label_dir)

    image_shape = [64, 64, 3]
    num_class = 33
    num_image = len(list_data)

    imdb = {'filename': list_data,
            'images': np.zeros([num_image] + image_shape),
            'labels': np.zeros([num_image] + image_shape[0:2])}

    for k in range(len(list_data)):
        print("processing image #%d: %s" % (k, list_data[k]))
        fn_data = os.path.join(data_dir, list_data[k])
        image = misc.imread(fn_data)
        if len(image.shape) == 2:
            image = np.tile(np.reshape(image, image.shape + [1]), [1, 1, 3])
        fn_label, _ = os.path.splitext(os.path.basename(fn_data))
        fn_label = os.path.join(label_dir, fn_label) + '.mat'
        # label = misc.imread(fn_label)
        S = sio.loadmat(fn_label)
        label = S['S']

        if not image_shape[0] == image.shape[0]:
            image = misc.imresize(arr=image, size=image_shape)
            label = misc.imresize(arr=label, size=image_shape[0:2], interp='nearest')

        imdb['images'][k] = image
        imdb['labels'][k] = label

    if is_save:
        dd.io.save(imdb_path, imdb)
    return imdb

def convertLabelMatToFiles(label_dir, outputDir, label_new_size = None):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    list_data = os.listdir(label_dir)
    for k in range(len(list_data)):
        print("processing label #%d: %s" % (k, list_data[k]))
        fn_label = os.path.join(label_dir, list_data[k])
        name, _ = os.path.splitext(os.path.basename(list_data[k]))
        S = sio.loadmat(fn_label)
        label = S['S']

        #if label_new_size is None or not label_new_size[0] == label.shape[0]:
            #label = misc.imresize(arr=label, size=label_new_size, interp='nearest')
        label = cv2.resize(label.astype('uint8'), dsize=(label_new_size[0], label_new_size[1]), interpolation=cv2.INTER_NEAREST)

        scipy.misc.imsave(outputDir + '/' + name + '.png', label)
        #cv2.imwrite(outputDir + '/' + name + '.png', label)
        #img = scipy.misc.toimage(label, high=np.max(label), low=np.min(label), mode='I')
        #img.save(outputDir + '/' + name + '.png')

def resizeData(data_dir, label_dir, output_dir, new_size = [64, 64, 3]):
    list_data = os.listdir(data_dir)
    num_image = len(list_data)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir + '/' + 'images'):
        os.makedirs(output_dir + '/' 'images')

    for k in range(len(list_data)):
        print("processing label #%d: %s" % (k, list_data[k]))
        fn_data = os.path.join(data_dir, list_data[k])
        name, _ = os.path.splitext(os.path.basename(list_data[k]))
        image = misc.imread(fn_data)
        if len(image.shape) == 2:
            image = np.tile(np.reshape(image, image.shape + [1]), [1, 1, 3])
        if not new_size[0] == image.shape[0]:
            #image = misc.imresize(arr=image, size=new_size[0:2] + [image.shape[2]])
            image = cv2.resize(image.astype('uint8'), dsize=(new_size[0], new_size[1]))
        scipy.misc.imsave(output_dir + '/' + 'images' + '/' + name + '.png', image)
    convertLabelMatToFiles(label_dir, output_dir + '/' + 'labels', new_size)

# SiftFlow database
data_dir = '/media/chinhan/dcnhan/Code/DAG-ResNet_new/data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories'
label_dir = '/media/chinhan/dcnhan/Code/DAG-ResNet_new/data/SiftFlowDataset/SemanticLabels/spatial_envelope_256x256_static_8outdoorcategories'
imdb_path = 'imdb_siftflow_img_label.h5'
#create_imdb_siftflowEx(data_dir, label_dir, imdb_path)
resizeData(data_dir, label_dir, './SiftFlow_Oct30', new_size=[256,256,3])
