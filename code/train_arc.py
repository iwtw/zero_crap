import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn

    
if __name__ == "__main__":

    config.train['split_random_seed'] = 0
    config.train['num_val_per_class'] = 10
    config.train['num_all_classes'] = 205
    config.train['num_zero_classes'] = 0
    config.train['num_test_classes'] = 45#fix

    config.train['optimizer'] == 'SGD'
    config.train['batch_size'] = 64
    config.train['lr_bounds'] = [ 0 , 40 , 60 ,72 ,  80  ] 
    config.train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]

    config.net['name'] = 'arc_resnet18'
    if 'gcn' in config.net:
        config.net.pop('gcn')
    if 'feature_net' in config.net:
        config.net.pop('feature_net')

    config.parse_config()
    config.net['type'] = 'all'
    for config.net['feature_layer_dim'] in [512]:
        print('================================================================================================================================================================================================================================')
        sleep(5)
        try:
            main(config)
        except KeyboardInterrupt:
            pass
