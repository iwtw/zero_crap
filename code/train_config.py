import torch.nn as nn
from functools import partial
#train_config

train = {}

train['split_random_seed'] = 0
train['num_val_per_class'] = 10
train['num_all_classes'] = 225
train['num_zero_classes'] = 30
train['num_test_classes'] = 65#fix

train['dataset'] = 'ZJL'
assert train['dataset'] in ['ZJL','msceleb','msceleb_origin','temp','test_speed','zero']
train['max_graph_hop'] = 5
train['graph_similarity'] = 'custom'
assert train['graph_similarity'] in ['custom','path','lch','wup']
train['graph_diagonal'] = 1

train['embedding_type'] = 'glove'
train['add_class_wordsembeddings'] = True
train['add_attributes'] = True
train['pca_dim'] =  None

train['attribute_index'] = None#this option is only valid when 'ris' 

train['batch_size'] = 64 
train['val_batch_size'] = 256
train['log_step'] = 10000000000
train['save_epoch'] = 1
train['save_metric'] = 'err'
train['optimizer'] = 'SGD'
train['learning_rate'] = 1e1

#-- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True 

train['mannual_learning_rate'] = True
#settings for mannual tuning
train['lr_bounds'] = [ 0 , 40 , 60 , 80 , 100 ]
train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]
    

#settings for auto tuning
train['use_cycle_lr'] = False
train['cycle_len'] = 1
train['num_cycles'] = 4
#train['num_restart'] = 5
train['cycle_mult'] = 1.6


#config for save , log and resume
train['resume'] = None
train['resume_epoch'] = None  #None means the last epoch
train['resume_optimizer'] = None
train['last_epoch'] = None





global net
net = {}
net['name'] = 'tede_resnet18'
net['input_shape'] = (64,64)
net['type'] = 'all'



loss = {}
loss['arcloss_start_epoch'] = 10
loss['weight_l2_reg'] = 5e-4
loss['weight_mse_attribute'] = 1.0
loss['m'] = 0.2
loss['s'] = 16
loss['T'] = 4
loss['weight_kl'] = loss['T'] ** 2
loss['weight_latent_feature'] = 1.0
#for idx ,v in enumerate(train['lrs']):
#    train['lrs'][idx] = v / loss['s'] 

test = {}
test['k'] = 3
test['pow_base'] = 0.8
test['delta'] = 1e-3




def parse_config():
    train['num_epochs'] = train['lr_bounds'][-1]
    split_args = [train['num_zero_classes'],train['num_val_per_class'],train['split_random_seed']]
    train['train_img_list'] = '../data/split_lists/{}_train_splitargs_{}_{}_{}.list'.format( train['dataset'] , *split_args )
    train['val_img_list'] = {'zero':'../data/split_lists/{}_zero_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args )  , 'non_zero':'../data/split_lists/{}_nonzero_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args ), 'all':'../data/split_lists/{}_all_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args ) }
    train['label_dict_file'] = '../data/split_lists/labelname_labelno_splitargs_{}_{}_{}.list'.format( *split_args )
    train['labelname_to_realname_file'] = '../data/label_list_wordnet.txt'
    embedding_file = '../data/split_lists/attributes_per_class_cleaned_plus_{}_class_wordembedding_sort_by_labelno_splitargs_{}_{}_{}.txt'.format( train['embedding_type'] , *split_args )

    train['class_attributes'] = embedding_file if train['add_class_wordsembeddings'] else '../data/split_lists/attributes_per_class_cleaned_sort_by_labelno_splitargs_{}_{}_{}.txt'.format( *split_args ) 
    train['attribute_file'] = train['class_attributes']
    train['novelty_train_img_list'] = '../data/split_lists/novelty_train_splitargs_{}_{}_{}.list'.format( *split_args  )
    train['novelty_val_img_list'] = '../data/split_lists/novelty_val_splitargs_{}_{}_{}.list'.format( *split_args )
    train['zero_class_file'] = '../data/split_lists/zero_classes_splitargs_{}_{}_{}'.format( *split_args )
    train['nonzero_class_file'] = '../data/split_lists/nonzero_classes_splitargs_{}_{}_{}'.format( *split_args )
    train['test_class_file'] = '../data/split_lists/test_classes_splitargs_{}_{}_{}'.format( *split_args )

    global net
    net_name = net['name']
    print('parsing----')
    print(net_name)
    net_type = net['type']
    net = {}
    net['name'] = net_name
    net['type'] = net_type
    net['input_shape'] = (64,64)
    net['num_attributes'] = 0 
    if  train['add_attributes']:
        net['num_attributes'] += 49
    if  train['add_class_wordsembeddings']:
        if train['pca_dim'] is not None:
            net['num_attributes'] += train['pca_dim']
        elif train['embedding_type'] == 'glove':
            net['num_attributes'] += 300
        elif train['embedding_type'] == 'elmo':
            net['num_attributes'] += 1024
    
    net['num_classes'] = train['num_all_classes'] - train['num_test_classes'] - train['num_zero_classes'] #40 test calsses
    if 'arc_resnet' in net['name']:
        net['strides'] = [1, 1, 2, 2, 1]#including first conv
        net['first_kernel_size'] = 3  
        net['fm_mult'] =  1.0
        net['use_batchnorm'] = True
        net['activation_fn'] = partial( nn.ReLU , inplace = True)
        net['pre_activation'] = True
        net['use_maxpool'] = False
        net['use_avgpool'] = True
        net['feature_layer_dim'] = 512
        net['dropout'] = 0.5
        #net['type'] = 'all'
        if net['type'] == 'coarse':
            net['num_classes'] = 10
            train['lr_bounds'] = [ 0 , 40 , 60 , 80 , 100 ]
            train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]
            net['feature_layer_dim'] = 128
    if 'rule_resnet' in net['name']:
        net['feature_layer_dim'] = net.pop('num_attributes')
        net['num_classes'] = 1
        net['strides'] = [2, 1, 2, 2, 1]#including first conv
        net['first_kernel_size'] = 3  
        net['fm_mult'] = 0.5
        net['use_batchnorm'] = True
        net['activation_fn'] = partial( nn.ReLU , inplace = True) 
        net['pre_activation'] = True
        net['use_maxpool'] = False
        net['use_avgpool'] = True
        net['dropout'] = 0.0
    if 'qfsl_resnet' in net['name']:
        net['num_classes'] = 1
        net['strides'] = [2, 1, 2, 2, 1]#including first conv
        net['first_kernel_size'] = 3  
        net['fm_mult'] = 1.0
        net['use_batchnorm'] = True
        net['activation_fn'] = partial( nn.ReLU , inplace = True) 
        net['pre_activation'] = True
        net['use_maxpool'] = False
        net['use_avgpool'] = True
        net['dropout'] = 0.5
        net['feature_layer_dim'] = 512
        net['visual_semantic_layers_kwargs'] = {}
        net['visual_semantic_layers_kwargs']['num_features'] = [512,512] 
        net['visual_semantic_layers_kwargs']['out_channels'] = net.pop('num_attributes')
        net['visual_semantic_layers_kwargs']['use_batchnorm'] = True
        net['visual_semantic_layers_kwargs']['activation_fn'] = partial( nn.LeakyReLU  , negative_slope = 0.2  )
        net['visual_semantic_layers_kwargs']['dropout'] = 0.0
    if 'ris_resnet' in net['name']:
        net.pop('num_attributes')
        net['num_classes'] = 1
        #net['strides'] = [1, 1, 2, 2, 1]#including first conv
        net['strides'] = [1, 1, 2 ]#including first conv
        net['first_kernel_size'] = 3  
        net['fm_mult'] = 0.5
        net['use_batchnorm'] = True
        net['activation_fn'] = partial( nn.ReLU , inplace = True)
        net['pre_activation'] = True
        net['use_maxpool'] = False
        net['use_avgpool'] = False
        net['feature_layer_dim'] = 1
        net['dropout'] = 0.0
    if 'gcn' in net['name']:
        net['feature_net'] = {}
        net['feature_net']['input_shape'] = (64,64)
        net['feature_net']['load_path'] = 'save/arc_resnet18_ZJL_shape-hw(64,64)_m0.2_s16_class_type-all_splitargs40_10_0_SGD/0'
        net['feature_net']['name'] = 'arc_resnet18'
        net['feature_net']['strides'] = [1, 1, 2, 2, 1]#including first conv
        net['feature_net']['first_kernel_size'] = 3  
        net['feature_net']['fm_mult'] =  1.0
        net['feature_net']['use_batchnorm'] = True
        net['feature_net']['activation_fn'] = partial( nn.ReLU , inplace = True)
        net['feature_net']['pre_activation'] = True
        net['feature_net']['use_maxpool'] = False
        net['feature_net']['use_avgpool'] = True
        net['feature_net']['feature_layer_dim'] = 128
        net['feature_net']['dropout'] = 0.5
        net['feature_net']['num_attributes'] = net['num_attributes']
        net['feature_net']['num_classes'] = train['num_all_classes'] - train['num_test_classes'] - train['num_zero_classes'] #40 test calsses
        net['gcn'] = {}
        net['gcn']['in_channels'] = net['num_attributes'] 
        net['gcn']['out_channels'] = net['feature_net']['feature_layer_dim']
        net['gcn']['num_features'] = [256,256,128]
        net['gcn']['fm_mult'] = 1.0
        net['gcn']['activation_fn'] = partial( nn.LeakyReLU , negative_slope = 0.2)
        net['gcn']['use_batchnorm'] = True

    if 'hse' in net['name']:
        train['save_metric'] = 'err_l2'
        train['lr_bounds'] = [ 0 ,10 ,  40 , 60 , 80 , 100 ]
        train['lrs'] = [ 1e-2 , 1e-3 , 1e-4 , 1e-5 ]
        num_classes = net.pop( 'num_classes' )
        net['resnet_kwargs'] = {}
        net['resnet_kwargs']['input_shape'] = net['input_shape']
        net['resnet_kwargs']['num_classes'] = 1
        net['resnet_kwargs']['strides'] = [1, 1, 2, 2, 1]#including first conv
        net['resnet_kwargs']['first_kernel_size'] = 3  
        net['resnet_kwargs']['fm_mult'] =  1.0
        net['resnet_kwargs']['use_batchnorm'] = True
        net['resnet_kwargs']['activation_fn'] = partial( nn.ReLU , inplace = True)
        net['resnet_kwargs']['pre_activation'] = True
        net['resnet_kwargs']['use_maxpool'] = False
        net['resnet_kwargs']['use_avgpool'] = True
        net['resnet_kwargs']['feature_layer_dim'] = None
        net['resnet_kwargs']['dropout'] = 0.5
        net['fm_mult'] = 0.5
        net['num_classes'] = [ 10 , num_classes ]
    else:
        train['save_metric'] = 'err'

    if 'tede' in net['name'] and 'resnet' in net['name']:
        net['feature_net_kwargs'] = {}
        net['feature_net_kwargs']['input_shape'] = net['input_shape']
        net['feature_net_kwargs']['strides'] = [1, 1, 2, 2, 1]#including first conv
        net['feature_net_kwargs']['first_kernel_size'] = 3  
        net['feature_net_kwargs']['fm_mult'] =  1.0
        net['feature_net_kwargs']['use_batchnorm'] = True
        net['feature_net_kwargs']['activation_fn'] = partial( nn.ReLU , inplace = True)
        net['feature_net_kwargs']['pre_activation'] = True
        net['feature_net_kwargs']['use_maxpool'] = False
        net['feature_net_kwargs']['use_avgpool'] = True
        net['feature_net_kwargs']['feature_layer_dim'] = None
        net['feature_net_kwargs']['dropout'] = 0.5
        net['feature_net_kwargs']['num_classes'] = 1 
        feature_layer_dim = 480
        net['visual_mlp_kwargs'] = {}
        net['visual_mlp_kwargs']['out_channels'] = feature_layer_dim
        net['visual_mlp_kwargs']['activation_fn'] = partial( nn.ReLU , inplace = True )
        net['visual_mlp_kwargs']['use_batchnorm'] = True
        net['semantic_mlp_kwargs'] = {}
        net['semantic_mlp_kwargs']['num_layers'] = 3
        net['semantic_mlp_kwargs']['out_channels'] = feature_layer_dim
        net['semantic_mlp_kwargs']['activation_fn'] = partial( nn.ReLU ,inplace = True )
        net['semantic_mlp_kwargs']['last_activation_fn'] = partial( nn.ReLU , inplace = True )
        net['semantic_mlp_kwargs']['use_batchnorm'] = True
    if 'gde' in net['name']:
        net['feature_net_kwargs'] = {}
        net['feature_net_kwargs']['input_shape'] = net['input_shape']
        net['feature_net_kwargs']['strides'] = [1, 1, 2, 2, 1]#including first conv
        net['feature_net_kwargs']['first_kernel_size'] = 3  
        net['feature_net_kwargs']['fm_mult'] =  1.0
        net['feature_net_kwargs']['use_batchnorm'] = True
        net['feature_net_kwargs']['activation_fn'] = partial( nn.ReLU , inplace = True)
        net['feature_net_kwargs']['pre_activation'] = True
        net['feature_net_kwargs']['use_maxpool'] = False
        net['feature_net_kwargs']['use_avgpool'] = True
        net['feature_net_kwargs']['feature_layer_dim'] = None
        net['feature_net_kwargs']['dropout'] = 0.5
        net['feature_net_kwargs']['num_classes'] = 1 
        feature_layer_dim = 640
        net['visual_mlp_kwargs'] = {}
        net['visual_mlp_kwargs']['out_channels'] = feature_layer_dim
        net['visual_mlp_kwargs']['activation_fn'] = partial( nn.ReLU , inplace = True )
        net['visual_mlp_kwargs']['use_batchnorm'] = True
        net['semantic_gconv_kwargs'] = {}
        net['semantic_gconv_kwargs']['out_channels'] = feature_layer_dim
        net['semantic_gconv_kwargs']['activation_fn'] = partial( nn.ReLU ,inplace = True )
        net['semantic_gconv_kwargs']['last_activation_fn'] = partial( nn.ReLU , inplace = True )
        net['semantic_gconv_kwargs']['use_batchnorm'] = True

    if 'tede' in net['name'] and 'nasnet' in net['name']:
        net['feature_net_kwargs'] = {}
        net['feature_net_kwargs']['input_shape'] = net['input_shape']
        #mobile
        '''
        net['feature_net_kwargs']['num_stem_features'] = 32
        net['feature_net_kwargs']['num_normal_cells'] = 4
        net['feature_net_kwargs']['filters'] = 44
        net['feature_net_kwargs']['scaling'] = 2
        '''
        #large
        net['feature_net_kwargs']['num_stem_features'] = 96
        net['feature_net_kwargs']['num_normal_cells'] = 5
        net['feature_net_kwargs']['filters'] = 128
        net['feature_net_kwargs']['scaling'] = 2

        net['feature_net_kwargs']['skip_reduction'] = False
        net['feature_net_kwargs']['use_aux'] = False
        net['feature_net_kwargs']['num_classes'] = 1
        feature_layer_dim = 480
        net['visual_mlp_kwargs'] = {}
        net['visual_mlp_kwargs']['out_channels'] = feature_layer_dim
        net['visual_mlp_kwargs']['activation_fn'] = partial( nn.ReLU , inplace = True )
        net['visual_mlp_kwargs']['use_batchnorm'] = True
        net['semantic_mlp_kwargs'] = {}
        net['semantic_mlp_kwargs']['out_channels'] = feature_layer_dim
        net['semantic_mlp_kwargs']['activation_fn'] = partial( nn.ReLU ,inplace = True )
        net['semantic_mlp_kwargs']['last_activation_fn'] = partial( nn.ReLU , inplace = True )
        net['semantic_mlp_kwargs']['use_batchnorm'] = True

        

    if 'arc' in net['name']:
        train['sub_dir'] = '{}_{}_shape-hw({},{})_m{:.1f}_s{}_class_type-{}_splitargs{}_{}_{}_{}'.format( net['name'] , train['dataset'],net['input_shape'][0],net['input_shape'][1] ,loss['m'] , loss['s'], net['type'] , *split_args , train['optimizer'] )
    elif 'gcn' in net['name']:
        train['sub_dir'] = '{}_{}_shape-hw({},{})_max-graph-hop{}_splitargs{}_{}_{}_{}'.format( net['name'] , train['dataset'],net['input_shape'][0],net['input_shape'][1] , train['max_graph_hop'] ,  *split_args ,train['optimizer'] )
    else:
        train['sub_dir'] = '{}_{}_shape-hw({},{})_splitargs{}_{}_{}_{}'.format( net['name'] , train['dataset'],net['input_shape'][0],net['input_shape'][1] , *split_args ,train['optimizer'] )

parse_config()
