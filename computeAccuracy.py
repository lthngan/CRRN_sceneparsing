import os
from scipy import misc
import numpy as np
from utils import accum

from itertools import product

nClasses = 34
#nClasses = 256
#prediction_path = './results/Nov02/Down64/SiftFlow_16_8_3_avg_newWeight_0.5_7000/labels_npy'
#gt_path = './results/Nov02/Down64/SiftFlow_16_8_3_avg_newWeight_0.5_7000/predictions_npy'
result_path = './results/Nov11/Down32/CamVid11_4_8_3_share'

gt_path = result_path + '/' + 'labels_npy'
#prediction_path = result_path + '/' + 'predict_DAG_max_npy'
prediction_path = result_path + '/' + 'predictions_npy'


files = os.listdir(prediction_path)

count = -1
confusion_all = None
meanAcc = 0
meanPixelAcc = 0
meanIntersectionAcc = 0
meanPixelAcc1 = 0
a = np.zeros((len(files),))
b = []
for f in files:
    # load ground truth and predict files
    fn_data_gt = os.path.join(gt_path, f)
    fn_data_pred = os.path.join(prediction_path, f)
    _, ext = os.path.splitext(os.path.basename(f))
    if ext == 'png':
        gt_label = misc.imread(fn_data_gt)
        pred_label = misc.imread(fn_data_pred)
    else:
        gt_label = np.load(fn_data_gt)
        pred_label = np.load(fn_data_pred)

    # compute statistics on accumulated pixels
    numPixels = np.sum(gt_label)
    arr = np.concatenate((gt_label.reshape((1, -1)), pred_label.reshape((1, -1))))
    arr = arr.T
    count = count + 1

    confusion = accum(arr, np.ones((arr.shape[0],)), size=[nClasses, nClasses])
    if confusion_all is None:
        confusion_all = confusion
    else:
        confusion_all += confusion

    # compute other statistics of the confusion matrix
    pos = np.sum(confusion, 1)
    res = np.sum(confusion, 0).T
    tp = np.diag(confusion)
    confusion1 = np.copy(confusion)
    confusion1[:,0] = 0
    confusion1[0,:] = 0
    tp1 = np.diag(confusion1)


    pixelAccuracy = np.sum(tp) / np.maximum(1, np.sum(confusion.flatten()))
    pixelAccuracy1 = np.sum(tp1) / np.maximum(1, np.sum(confusion1.flatten()))

    meanAccuracy = np.mean(tp / np.maximum(1, pos))
    meanIntersectionUnion = np.mean(tp / np.maximum(1, pos + res - tp))

    meanAcc += meanAccuracy
    meanPixelAcc += pixelAccuracy
    meanPixelAcc1 += pixelAccuracy1
    a[count] = pixelAccuracy1
    b.append(f)
    meanIntersectionAcc += meanIntersectionUnion

meanAcc /= count
meanPixelAcc /= count
meanPixelAcc1 /= count
meanIntersectionAcc /= count

print('Mean Pixel Accuracy over %d testing images: %f' % (count, meanPixelAcc))
print('Mean Pixel Accuracy (without background class) over %d testing images: %f' % (count, meanPixelAcc1))
print('Mean Accuracy over %d testing images: %f' % (count, meanAcc))
print('Mean Intersection Accuracy over %d testing images: %f' % (count, meanIntersectionAcc))
