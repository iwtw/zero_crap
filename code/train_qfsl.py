import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn

    
if __name__ == "__main__":
    config.train['split_random_seed'] = 0
    config.train['num_val_per_class'] = 10
    config.train['num_all_classes'] = 230
    #config.train['num_zero_classes'] = 40
    config.train['num_test_classes'] = 40#fix
    config.train['max_graph_hop'] = 4


    config.train['optimizer'] == 'SGD'
    config.train['batch_size'] = 64
    config.train['num_epochs'] = 100
    config.train['lr_bounds'] = [ 0 , 40 , 60 , 80 , config.train['num_epochs'] ]
    config.train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]

    config.net['name'] = 'qfsl_resnet18'
    if 'gcn' in config.net:
        config.net.pop('gcn')
    if 'feature_net' in config.net:
        config.net.pop('feature_net')

    for config.train['num_zero_classes']  in [40]:
        for config.train['pca_dim'] in [128]:
            config.parse_config()
            config.net['type'] = 'all'
            for config.net['feature_layer_dim'] in [ 512 ]:
                for config.net['visual_semantic_layers_kwargs']['dropout'] in [0,0.5]:
                    print('================================================================================================================================================================================================================================')
                    sleep(5)
                    try:
                        main(config)
                    except KeyboardInterrupt:
                        pass
