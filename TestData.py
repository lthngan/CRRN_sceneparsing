import image_processing
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy
from scipy import misc
from skimage.color import label2rgb

filename_queue = tf.train.string_input_producer(['./SiftFlow_Oct30_new/data_train/train-00000-of-00001'])
reader = tf.TFRecordReader()
_, value = reader.read(filename_queue)
feature_map = {
      'image/label_encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/filename':tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),

#      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
#                                              default_value=-1),
#      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
#                                             default_value=''),
}
sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
# Sparse features in Example proto.
# feature_map.update(
#     {k: sparse_float32 for k in ['image/object/bbox/xmin',
#                                  'image/object/bbox/ymin',
#                                  'image/object/bbox/xmax',
#                                  'image/object/bbox/ymax']})

features = tf.parse_single_example(value, feature_map)
#label = tf.cast(features['image/class/label'], dtype=tf.int32)


a = features['image/encoded']
b = features['image/label_encoded']
c = features['image/filename']

my_img = tf.image.decode_png(a)
my_label = tf.image.decode_png(b)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1000): #length of your filename list
    #image = my_img.eval() #here is your image Tensor :)
#    label = my_label.eval()
    #name = c.eval()
    image, label, name = sess.run([my_img, my_label, c])
    a =np.max(label)
    #if a == 33:
    #    print name

#    img_feat = features['image/encoded'].eval()

    #Image.fromarray(np.asarray(image)).save('temp/test%d.jpg'%i)
    #misc.imsave('temp/img_%s'%name, image)
    #misc.imsave('temp/label_%s'%name, label[:,:,0])
    misc.imsave('temp/combine_%s'%name, label2rgb(label[:,:,0], image=image))

  coord.request_stop()
  coord.join(threads)