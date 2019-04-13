import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn
from tqdm import tqdm
import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name',type=str,choices=['gcn','tede'],default='tede')
    parser.add_argument('-num_models',type=int,default=10)
    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_info = '../data/bagging_results/' + time + '.txt'
    parser.add_argument('-output_list',default = model_info )
    parser.add_argument('--start',type=int,default=0)
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    cmd = 'mkdir -p {}'.format( '/'.join( args.output_list.split('/')[:-1] ))
    print(cmd)
    os.system(cmd )

    log = []
    for config.train['split_random_seed'] in tqdm(range(args.start,args.num_models) , desc='bagging' , leave=False):

        if args.model_name == 'tede':
            config.net['name'] = 'tede_resnet18'
            config.train['optimizer'] = 'SGD'
            config.train['batch_size'] = 64
            #config.train['lr_bounds'] = [ 0 , 40 , 60 ,72 ,  80 ]  
            #config.train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4  ]
            config.train['lr_bounds'] = [ 0 , 40 ,  48 ,  52 ] 
            config.train['lrs'] = [ 1e-1 , 1e-2 , 1e-3  ] 
            config.train['add_attributes'] = True
            config.train['add_class_wordsembeddings'] = True

            config.parse_config()
            activation_fn = partial( nn.LeakyReLU )
            config.net['semantic_mlp_kwargs']['last_activation_fn'] = activation_fn
            config.net['visual_mlp_kwargs']['activation_fn'] = activation_fn
            feature_layer_dim = 384
            config.net['visual_mlp_kwargs']['out_channels'] = feature_layer_dim
            config.net['semantic_mlp_kwargs']['out_channels'] = feature_layer_dim
            result = main( config )
            log.append( [ str(config.train['split_random_seed'])  , result['log_path'] , result['log_path'], str(1 - result['non_zero']) , str(1 - result['zero']) ] )
        elif args.model_name == 'gcn':
            config.train['save_metric'] = 'err'

            config.train['optimizer'] = 'SGD'
            config.train['batch_size'] = 64
            config.train['lr_bounds'] = [ 0 , 40 , 60 ,72 ,  80 ]  
            config.train['lrs'] = [ 3e-1 , 3e-2 , 3e-3 , 1e-4  ]
            #config.train['lr_bounds'] = [0,1]
            #config.train['lrs'] = [1e-1]
            config.train['add_class_wordsembeddings'] = True
            config.train['add_attributes'] = True
            config.net['name'] = 'arc_resnet18'
            config.net['type'] = 'all'
            config.parse_config()
            config.net['fm_mult'] = 1.0
            train_results1 = main(config)

            config.net['name'] = 'gcn'
            config.train['batch_size'] = 512
            config.train['optimizer'] = 'Adam'
            config.train['lr_bounds'] = [ 0  , 12 , 25 , 40 , 55 ]
            config.train['lrs'] = [ 1e-2 , 1e-3 , 1e-4 , 1e-5 ]
            #config.train['lr_bounds'] = [0,1]
            #config.train['lrs'] = [1e-1]
            config.train['add_class_wordsembeddings'] = True
            config.train['add_attributes'] = True
            config.train['pca_dim'] = None
            config.train['max_graph_hop'] = 5
            config.train['graph_diagonal'] = 4
            config.train['graph_similarity'] = 'custom'
            config.parse_config()
            config.train['save_metric'] = 'err'
            config.net['gcn']['fm_mult'] = 1.0
            config.net['gcn']['num_features'] = [256,256,128]
            config.net['feature_net']['load_path'] = train_results1['log_path']
            train_results2 = main(config)
            log.append( [ str(config.train['split_random_seed'])  , train_results1['log_path'] , train_results2['log_path'], str(1 - train_results1['non_zero']) , str(1 - train_results2['zero']) ] )
        with open(args.output_list,'w') as fp:
            write_msg = '\n'.join( [ " ".join(v) for v in log ] ) 
            fp.write(write_msg+'\n')
            fp.flush()
    print( log )



        
