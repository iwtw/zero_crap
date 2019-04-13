import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn


    
if __name__ == "__main__":

    config.net['name'] = 'tede_resnet18'
    config.train['batch_size'] = 64
    config.train['split_random_seed'] = 0
    
    config.train['lr_bounds'] = [ 0  , 40 , 48 , 52 , 56 ]  
    config.train['lrs'] = [1e-1,1e-2,1e-3,1e-4]
    
    config.parse_config()
    best_zero = 1e9
    best_result = None
    for feature_layer_dim in [384,480,560]:
        for config.net['semantic_mlp_kwargs']['num_layers'] in [3]:
            config.net['visual_mlp_kwargs']['out_channels'] = feature_layer_dim
            config.net['semantic_mlp_kwargs']['out_channels'] = feature_layer_dim
            for config.loss['weight_latent_feature'] in [1]:
                for activation_fn in [partial( nn.LeakyReLU )]:
                    config.net['semantic_mlp_kwargs']['last_activation_fn'] = activation_fn
                    config.net['visual_mlp_kwargs']['activation_fn'] = activation_fn
                    print('================================================================================================================================================================================================================================')
                    sleep(5)
                    try:
                        result = main(config)
                        if best_zero > result['zero']:
                            best_zero = result['zero']
                            best_result = result
                        
                    except KeyboardInterrupt:
                        pass
    print('best result :')
    print( best_result )
