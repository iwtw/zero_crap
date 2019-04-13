import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn
from custom_utils import *


    
if __name__ == "__main__":
    config.train['num_zero_classes'] = 40
    if 'gcn' in config.net:
        config.net.pop('gcn')
    if 'feature_net' in config.net:
        config.net.pop('feature_net')

    config.net['name'] = 'arc_resnet50'
    config.net['type'] = 'coarse'
    config.train['pca_dim']  = 230
    config.parse_config()
    for config.net['feature_layer_dim'] in [128,256]:
        print('================================================================================================================================================================================================================================')
        sleep(10)
        try:
            main(config)
        except KeyboardInterrupt:
            pass
