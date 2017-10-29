import numpy as np
from skimage.color import label2rgb
import scipy
from scipy import misc

COLORS = (
'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'gray',
'green', 'greenyellow', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue',
'mediumspringgreen', 'mediumvioletred', 'midnightblue', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal',
'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen')

nClasses = 34
W = np.ceil(np.sqrt(nClasses)).astype(np.int32)
H = np.ceil(1.0 * nClasses / W).astype(np.int32)
block_size = 32

count = -1
label = np.zeros((H * block_size,W * block_size), dtype=np.int32) - 1
for r in range(H):
    for c in range(W):
        count = count + 1
        label[r * block_size: (r + 1) *block_size, c * block_size : (c +1) * block_size] = count

img = np.zeros((label.shape[0], label.shape[1],3), dtype=np.int32)
img[:, :, 0] = 255
img[:, :, 1] = 0
img[:, :, 2] = 0
class_img = label2rgb(label, image=img, colors=COLORS)
misc.imsave('class_img2.png', class_img)




