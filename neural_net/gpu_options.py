#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:23:14 2022
History:
04/15/2022: 

@author: jaerock
"""

import keras.backend as K
import tensorflow as tf

def set():
    # to address the error:
    #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.tensorflow_backend.set_session(sess)