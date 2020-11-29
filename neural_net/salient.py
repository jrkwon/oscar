#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 12:28 2019

@author: doshininad
"""

"""
**** NOTE NOTE NOTE ****
This module requires the kears-vis package.
This module is not tested. 
Must be fixed to use model path as an argument.
"""

import numpy as np
from matplotlib import pyplot as plt
from net_model import NetModel
from config import Config
#from keras.preprocessing.image import img_to_array
import sys
import os
sys.path.append(str(os.environ['HOME']) + ('/keras-vis'))
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay
import cv2
from scipy import misc

model_path = '/home/ghor9797/NCD_Github/python/pretrained_weights/MirNet_C_4/2019-02-28-17-31-47'
config = Config()
net_model = NetModel(model_path)   
net_model.load()
img = utils.load_img('/home/ghor9797/NCD_Github/test/2019-02-28-17-31-49-300752.jpg', target_size=(config.image_size[1], config.image_size[0]))
print(img.shape)
#cv2.imshow('image',img)
misc.imsave('original.jpg', img)

# Convert to BGR, create input with batch_size: 1.
#bgr_img = utils.bgr2rgb(img)
img_input = np.expand_dims(img, axis=0)
pred = net_model.model.predict(img_input)[0][0]
print('Predicted {}'.format(pred))

titles = ['right steering', 'left steering', 'maintain steering']
modifiers = [None, 'negate', 'small_values']
for i, modifier in enumerate(modifiers):
    heatmap = visualize_saliency(net_model.model, layer_idx=-1, filter_indices=0, seed_input=img, grad_modifier=modifier, keepdims=True)
    print(heatmap.shape)
    misc.imsave('heatmap_%02d.jpg' % i, heatmap)
    #plt.figure()
    #cv2.imshow('heatmap',heatmap)
    #plt.title(titles[i])
    # Overlay is used to alpha blend heatmap onto img.
    #plt.imshow(overlay(img, heatmap, alpha=0.7))
    final = overlay(img, heatmap, alpha=0.5)
    misc.imsave('final_%02d.jpg' % i, final)
