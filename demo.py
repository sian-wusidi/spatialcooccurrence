#!/usr/bin/env python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from UNETGAN.models import create_models, build_graph


c,inte,dis = create_models(n_channels=3)
a = build_graph(c, inte) 
c.load_weights('weights/classifier_10.hdf')
a.load_weights('weights/attention_10.hdf')


img_unlab = np.load('Datasamples//paired//10_sheet_1011_1956.npz')
img_lab = np.load('Datasamples//paired//10_sheet_12_1883.npz')
GT_lab = np.load('Datasamples//paired//10_annotation_12_1883.npz')
img_unlab = img_unlab['arr_0']
img_unlab = img_unlab.reshape(1, 256, 256, 3)
img_lab = img_lab['arr_0']
img_lab = img_lab.reshape(1, 256, 256, 3)
GT_lab = GT_lab['arr_0']
GT_lab = GT_lab.reshape(1, 256, 256, 4)


pred_lab = c.predict(img_lab)
pred_unlab = c.predict(img_unlab)
pred_mask = a.predict([img_lab, GT_lab, img_unlab])
plt.subplot(231)
plt.imshow(img_lab.squeeze())

plt.subplot(232)
plt.imshow(img_unlab.squeeze())

plt.subplot(233)
plt.imshow(GT_lab[:,:,:,0:3].squeeze())

plt.subplot(234)
plt.imshow(pred_lab[:,:,:,0:3].squeeze())

plt.subplot(235)
plt.imshow(pred_unlab[:,:,:,0:3].squeeze())

plt.subplot(236)
plt.imshow(pred_mask.squeeze())


plt.show()