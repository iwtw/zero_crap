import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn


    
if __name__ == "__main__":

    config.net['name'] = 'gcn'
    config.train['batch_size'] = 512
    config.train['num_zero_classes'] = 30
    config.train['split_random_seed'] = 5
    '''
    config.train['resume'] = 'save/gcn_ZJL_shape-hw(64,64)_max-graph-hop7_splitargs40_10_0_Adam/0'
    config.train['resume_epoch'] = 'best_non_zero'
    config.train['resume_optimizer'] = 'save/gcn_ZJL_shape-hw(64,64)_max-graph-hop7_splitargs40_10_0_Adam/0'
    config.train['last_epoch'] = 159
    '''

    '''
    config.train['lr_bounds'] = [ 0  ,10 , 40 , 70, 100 , 130 ]  
    config.train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 , 1e-5 ]
    '''

    '''
    config.train['optimizer'] = 'SGD'
    #for SGD
    config.train['optimizer'] = 'SGD'
    config.train['lr_bounds'] = [ 0  , 10 , 30 , 50 , 70 ]
    config.train['lrs'] = [ 1e-0 , 1e-1 , 1e-2 , 1e-3 ]
    '''
    #for Adam
    config.train['optimizer'] = 'Adam'
    config.train['lr_bounds'] = [ 0  , 12 , 30 , 50 , 70 ]
    config.train['lrs'] = [ 1e-2 , 1e-3 , 1e-4 , 1e-5 ]
    '''
    config.train['use_cycle_lr'] = True
    config.train['cycle_lrs'] = [1e-2,5e-3,1e-3,1e-3,1e-3,5e-4,1e-4,1e-4]
    config.train['cycle_bounds'] = [0,1,3,5,10,15,25,45,100]
    '''

    
    for config.train['graph_diagonal'] in [4]:
        for config.train['graph_similarity'] in ['custom','path','lch','wup','custom']:
            for config.train['max_graph_hop']  in [5]: 
                for config.train['pca_dim'] in [None]:
                    config.parse_config()
                    config.net['feature_net']['load_path'] = 'save/arc_resnet18_ZJL_shape-hw(64,64)_m0.2_s16_class_type-all_splitargs30_10_5_SGD/0'
                    for config.net['gcn']['num_features'] in [ [256,256,128] ]:
                        config.net['gcn']['fm_mult'] = 1.0
                        print('================================================================================================================================================================================================================================')
                        sleep(5)
                        try:
                            main(config)
                        except KeyboardInterrupt:
                            pass
