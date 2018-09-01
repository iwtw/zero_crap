import torch.nn as nn
from functools import partial
#train_config

train = {}
train['dataset'] = 'zero'
assert train['dataset'] in ['msceleb','msceleb_origin','temp','test_speed','zero']
train['train_img_list'] = 'data/{}_train.list'.format( train['dataset']  )
train['val_img_list'] = {'zero':'data/{}_zero_val.list'.format( train['dataset'])  , 'non_zero':'data/{}_multi_val.list'.format( train['dataset']), 'all':'data/{}_all_val.list'.format( train['dataset']) }
#train['val_img_list'] = {'non_zero':'data/{}_multi_val.list'.format( train['dataset'])}
train['label_dict_file'] = './data/labelname_labelno.list'
train['labelname_to_realname_file'] = './data/DatasetA_train_20180813/label_list_wordnet.txt'
train['max_graph_hop'] = 17


train['add_class_wordsembeddings'] = True
train['class_attributes'] = './data/attributes_per_class_cleaned_plus_class_wordembedding_sort_by_labelno.txt' if train['add_class_wordsembeddings'] else './data/attributes_per_class_cleaned_sort_by_labelno.txt'
train['attribute_file'] = train['class_attributes']
train['attribute_index'] = None#this option is only valid when 'ris' 

train['batch_size'] = 1 
train['val_batch_size'] = 256
train['num_epochs'] = 3000
train['log_step'] = 10000000000
train['save_epoch'] = 1
train['save_metric'] = 'err'
train['optimizer'] = 'Adam'
train['learning_rate'] = 1e1

#-- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True 

train['mannual_learning_rate'] = True
#settings for mannual tuning
train['lr_bounds'] = [ 0 , 60 , 100 , 3000 , train['num_epochs'] ]
train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-3 ]

#settings for auto tuning
train['cycle_len'] = 1
train['num_cycles'] = 4
#train['num_restart'] = 5
train['cycle_mult'] = 1.6


#config for save , log and resume
train['resume'] = None
train['resume_epoch'] = None  #None means the last epoch
train['resume_optimizer'] = None




net = {}
net['name'] = 'gcn'
net['input_shape'] = (64,64)
net['num_attributes'] = 326  if train['add_class_wordsembeddings'] else 26
net['num_classes'] = 230 - 60
if 'arc_resnet' in net['name']:
    net['strides'] = [1, 1, 2, 2, 1]#including first conv
    net['first_kernel_size'] = 3  
    net['fm_mult'] =  1.0
    net['use_batchnorm'] = True
    net['activation_fn'] = partial( nn.ReLU , inplace = True)
    net['pre_activation'] = True
    net['use_maxpool'] = False
    net['use_avgpool'] = True
    net['feature_layer_dim'] = 128
    net['dropout'] = 0.5
if 'rule_resnet' in net['name']:
    net['num_classes'] = 1
    net['strides'] = [2, 1, 2, 2, 1]#including first conv
    net['first_kernel_size'] = 3  
    net['fm_mult'] = 0.5
    net['use_batchnorm'] = True
    net['activation_fn'] = partial( nn.ReLU , inplace = True) 
    net['pre_activation'] = True
    net['use_maxpool'] = False
    net['use_avgpool'] = True
    net['feature_layer_dim'] = net.pop('num_attributes')
    net['dropout'] = 0.0
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
    net['feature_net']['num_attributes'] = 326  if train['add_class_wordsembeddings'] else 26
    net['feature_net']['num_classes'] = 230 - 60
    net['feature_net']['load_path'] = './save/arc_resnet18_zero_shape-hw(64,64)_m0.2_s16_SGD/124'
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
    net['gcn'] = {}
    net['gcn']['in_channels'] = net['num_attributes'] 
    net['gcn']['num_features'] = [512,256]
    net['gcn']['out_channels'] = net['feature_net']['feature_layer_dim']
    net['gcn']['fm_mult'] =1.0 
    net['gcn']['activation_fn'] = partial( nn.ReLU , inplace = True)
    net['gcn']['use_batchnorm'] = False


loss = {}
loss['arcloss_start_epoch'] = 1000000
loss['weight_l2_reg'] = 5e-4
loss['weight_mse_attribute'] = 1.0
loss['m'] = 0.2
loss['s'] = 16
#for idx ,v in enumerate(train['lrs']):
#    train['lrs'][idx] = v / loss['s'] 

test = {}
test['k'] = 10
test['pow_base'] = 0.8
test['delta'] = 1e-3


if 'arc' in net['name']:
    train['sub_dir'] = '{}_{}_shape-hw({},{})_m{:.1f}_s{}_{}'.format( net['name'] , train['dataset'],net['input_shape'][0],net['input_shape'][1] ,loss['m'] , loss['s'],  train['optimizer'] )
else:
    train['sub_dir'] = '{}_{}_shape-hw({},{})_{}'.format( net['name'] , train['dataset'],net['input_shape'][0],net['input_shape'][1] , train['optimizer'] )

