import torch.nn as nn
from functools import partial
#train_config

train = {}
train['dataset'] = 'zero'
assert train['dataset'] in ['msceleb','msceleb_origin','temp','test_speed','zero']
train['train_img_list'] = 'data/{}_train.list'.format( train['dataset']  )
train['val_img_list'] = {'zero':'data/{}_zero_val.list'.format( train['dataset']) , 'all':'data/{}_all_val.list'.format( train['dataset']) , 'multi':'data/{}_multi_val.list'.format( train['dataset'])}
train['attribute_file'] = './data/DatasetA_train_20180813/attributes_per_class_cleaned.txt'
train['label_dict_file'] = './data/labelname_labelno.list'

train['batch_size'] = 64 
train['val_batch_size'] = 64
train['num_epochs'] = 140
train['log_step'] = 10
train['save_epoch'] = 1
train['optimizer'] = 'SGD'
train['learning_rate'] = 1e-1

#-- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True 

train['mannual_learning_rate'] = True
#settings for mannual tuning
train['lr_bounds'] = [ 0 , 60 , 100 , 120 , train['num_epochs'] ]
train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]

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
net['name'] = 'arc_resnet18'
net['input_shape'] = (64,64)
net['num_classes'] = 230 - 50
net['num_attributes'] = 26
if 'resnet' in net['name']:
    net['strides'] = [2, 1, 2, 2, 1]#including first conv
    net['first_kernel_size'] = 3  
    net['fm_mult'] = 0.5
    net['use_batchnorm'] = True
    net['activation_fn'] = partial( nn.ReLU , inplace = True)
    net['pre_activation'] = True
    net['use_maxpool'] = False
    net['use_avgpool'] = False
    net['feature_layer_dim'] = 128
    net['dropout'] = 0.5





loss = {}
loss['arcloss_start_epoch'] = 20
loss['weight_l2_reg'] = 1e-4
loss['m'] = 0.2
loss['s'] = 16
#for idx ,v in enumerate(train['lrs']):
#    train['lrs'][idx] = v / loss['s'] 

test = {}
test['k'] = 10
test['pow_base'] = 0.8
test['delta'] = 1e-3


train['sub_dir'] = '{}_{}_shape-hw({},{})_m{:.1f}_s{}_{}'.format( net['name'] , train['dataset'],net['input_shape'][0],net['input_shape'][1] ,loss['m'] , loss['s'],  train['optimizer'] )
