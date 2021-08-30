#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: donghyun
"""
import sys
import net_model
import const
from config import Config
from keras.utils.vis_utils import plot_model as vs_plot_model
from keras.utils import plot_model as plot_model
###############################################################################
#
# --install--
# pip install pyparsing
# pip install graphviz
# pip install pydot
# conda install graphviz

def model_list():
    if config['network_type'] == const.NET_TYPE_PILOT_M:
        model = net_model.model_pilotnet_m()
    elif config['network_type'] == const.NET_TYPE_PILOT:
        model = net_model.model_pilotnet()
    elif config['network_type'] == const.NET_TYPE_VS:
        model = net_model.model_vs()
    elif config['network_type'] == const.NET_TYPE_DAVE2SKY:
        model = net_model.model_dave2sky()
    elif config['network_type'] == const.NET_TYPE_VGG16:
        model = net_model.model_vgg16()
    elif config['network_type'] == const.NET_TYPE_ALEX:
        model = net_model.model_alexnet()
    elif config['network_type'] == const.NET_TYPE_RES:
        model = net_model.model_resnet18()
    elif config['network_type'] == const.NET_TYPE_ALEX_M:
        model = net_model.model_alexnet_m()
    elif config['network_type'] == const.NET_TYPE_ALEX_T:
        model = net_model.model_alexnet_t()
    elif config['network_type'] == const.NET_TYPE_CONJOIN_T:
        model = net_model.model_conjoin_t()
    elif config['network_type'] == const.NET_TYPE_CONJOIN:
        model = net_model.model_conjoin()
    elif config['network_type'] == const.NET_TYPE_PILOTwL:
        model = net_model.model_pilotnet_lstm()
    elif config['network_type'] == const.NET_TYPE_ALEXwL_T:
        model = net_model.model_alexnet_t_lstm()
    else:
        exit('ERROR: Invalid neural network type.')
    return model

config = Config.neural_net
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python {} save_path'.format(sys.argv[0]))

        # main(sys.argv[1], sys.argv[2])
        model = model_list()
        vs_plot_model(model, to_file=str(sys.argv[1])+'N'+str(config['network_type'])+'.png', show_shapes=True, show_layer_names=False)

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
