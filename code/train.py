import torch
import torch.nn as nn
import numpy as np
from log import TensorBoardX,LogParser
from utils import *
from dataset import *
from tqdm import tqdm
from time import time
from network import *
from custom_utils import *
import sys
from functools import partial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn
from copy import deepcopy
import torch.nn.functional  as F
import tensorflow as tf


def init(config):
    os.system('python ./preprocess_data/split.py -num_zero_classes {} -num_val_per_class {} -random_seed {}'.format( config.train['num_zero_classes'] , config.train['num_val_per_class'] , config.train['split_random_seed'] ))
    m = config.loss['m']
    global cos_m , sin_m , border  , train_num_classes 
    cos_m = np.cos( m )
    sin_m = np.sin( m )
    border = np.cos( math.pi - m )
    #train_dataset.class_attributes = np.loadtxt( config.train['train_dataset.class_attributes'] )
    #if not config.train['add_attributes']:
    #    train_dataset.class_attributes = train_dataset.class_attributes[:,26:]
    train_list = open(config.train['train_img_list']).read().strip().split('\n')
    label_dict = {}
    for line in train_list:
        if line.split('\t')[1] not in label_dict:
            label_dict[ line.split('\t')[1] ] = 1
    train_num_classes = len(label_dict)
    print( 'train_num_classes : {}'.format( train_num_classes ) )




def mannual_learning_rate( optimizer , epoch ,  step , num_step_epoch , config ):
    
    if not config.train['use_cycle_lr']:
        bounds = config.train['lr_bounds']
        lrs = config.train['lrs']
        for idx in range(len(bounds) - 1):
            if bounds[idx] <= epoch and epoch < bounds[idx+1]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrs[idx]
    else:
        bounds = config.train['cycle_bounds']
        lrs = config.train['cycle_lrs']
        for idx in range(len(bounds) - 1):
            if bounds[idx] <= epoch and epoch < bounds[idx+1]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrs[idx]
                break
        for param_group in optimizer.param_groups:
            length = bounds[idx+1] -  bounds[idx]
            param_group['lr'] *= np.cos( np.pi / 2 / (length * num_step_epoch) * (step + num_step_epoch * ( epoch - bounds[idx] )) )


        




def compute_loss1( results , batch , epoch , config ,  class_range , is_training = False , mse_attribute = True ):
    s, m, k = config.loss['s'] , config.loss['m'] , config.test['k']
    loss_dict = {}
    #print(results['fc'].shape ,  batch['label'].max() )
    if config.net['type'] == 'coarse':
        labels = batch['super_class_label']
    else:
        labels = batch['label']

    if is_training:
        sum_exp = torch.exp(results['fc']).sum(1)
        loss_dict['err'] = 1 - torch.eq( labels  , torch.max( results['fc'] , 1 )[1] ).float().mean()
        topk , top_idx = torch.topk( torch.exp(results['fc'] ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #tqdm.write( "{}".format(topk[0]), file=sys.stdout )
        if epoch < config.loss['arcloss_start_epoch'] :
            loss_dict['softmax'] = cross_entropy(  s*results['fc'] , labels )

        else:
            cos_theta = results['fc']
            cos_theta_yi = cos_theta[( torch.arange(cos_theta.shape[0] , dtype = torch.int64 ) , labels )  ]
            sin_theta_yi = ( 1 - cos_theta_yi**2 ) **0.5
            phai_theta_yi = cos_theta_yi * cos_m - sin_theta_yi * sin_m
            phai_theta_yi = torch.where( cos_theta_yi > border , phai_theta_yi , -2. - phai_theta_yi )#the loss curve correction
            phai_theta = cos_theta
            phai_theta[ ( torch.arange( cos_theta.shape[0] , dtype = torch.int64 ) , labels ) ] = phai_theta_yi
            loss_dict['aam'] = cross_entropy( s * phai_theta , labels )
    else:
        k = config.test['k']
        fc = results['fc']
        sum_exp = torch.exp(fc).sum(1)
        #topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #for i in range(k):
        #    loss_dict['top{}'.format(i+1)] = torch.mean( topk[:,i] )

        
        #top_idx = top_idx.cpu().detach().numpy()

        predicts = get_predict( results , config , train_dataset.class_attributes , class_range )
        loss_dict['err'] = 1 - torch.eq( predicts , labels ).float().mean() 

    if mse_attribute:
        loss_dict['mse_attribute'] = mse( results['attribute'] , batch['attribute'] )
        #loss_dict['mse_feature_'] = mse( results['feature_'] , results['feature'].detach() )
    return loss_dict

def compute_loss2( results , batch , epoch , config , class_range , is_training = False ):
    loss_dict = {}
    loss_dict['mse_attribute'] = mse( results['attribute'] , batch['attribute'] )
    attr_list_tensor = torch.FloatTensor( train_dataset.class_attributes ).cuda()
    dis = torch.norm( results['attribute'].view( results['attribute'].shape[0] , 1 , -1 ) - attr_list_tensor.view( 1 , attr_list_tensor.shape[0] , -1 ) , p = 2 , dim = 2 )
    predicts = torch.min( dis, dim = 1 )[1]
    loss_dict['err'] = 1 -  torch.eq( predicts , batch['label'] ).float().mean() 
    return loss_dict

def compute_loss3( results , batch , epoch , config , class_range , is_training = False ):

    loss_dict = {}
    if not is_training:
        predicts = get_predict( results , config , train_dataset.class_attributes , class_range = class_range )
        test_predicts = get_predict( results , config , train_dataset.class_attributes , class_range = (config.train['num_all_classes'] - config.train['num_zero_classes'] - config.train['num_test_classes'] , config.train['num_all_classes'] ) )
        all_predicts = get_predict( results , config , train_dataset.class_attributes , class_range = (0,config.train['num_all_classes']) )
        feature_net_predicts = get_predict( {'fc':results['feature_net_fc']} , config , train_dataset.class_attributes )
        border = config.net['feature_net']['num_classes']
        mix_fc = torch.cat( [results['feature_net_fc'][:,:border] , results['fc'][:,border:]] , dim = 1 )
        #print(mix_fc.shape)
        mix_predicts = get_predict( {'fc':mix_fc}, config , train_dataset.class_attributes )

        loss_dict['err'] = 1 - torch.eq( predicts , batch['label'] ).float().mean() 
        loss_dict['feature_net_err'] = 1 - torch.eq( feature_net_predicts , batch['label'] ).float().mean() 
        loss_dict['mix_err'] = 1 - torch.eq( mix_predicts , batch['label'] ).float().mean() 
        loss_dict['all_err'] =  1 - torch.eq( all_predicts , batch['label'] ).float().mean() 
        loss_dict['test_err'] = 1 - torch.eq( test_predicts , batch['label'] ).float().mean() 

        sum_exp = torch.exp( results['fc']).sum(1)
        topk , top_idx = torch.topk( torch.exp( results['fc'] ) / sum_exp.view(sum_exp.shape[0],1) , k = config.test['k'] , dim = 1 )
        for i in range(config.test['k']):
            loss_dict['top{}'.format(i+1)] = torch.mean( topk[:,i] )

        sum_exp = torch.exp( results['feature_net_fc']).sum(1)
        topk , top_idx = torch.topk( torch.exp( results['feature_net_fc'] ) / sum_exp.view(sum_exp.shape[0],1) , k = config.test['k'] , dim = 1 )
        for i in range(config.test['k']):
            loss_dict['fnet_top{}'.format(i+1)] = torch.mean( topk[:,i] )

    loss_dict['mse_weight'] =  mse( results['weight'][:train_num_classes] ,  F.normalize(label_weight) )
    return loss_dict

def compute_loss4( results , batch , epoch , config , class_range , is_training = False ):

    results['fc'] = results['s1'].clone()
    old_type = config.net['type']
    config.net['type'] = 'coarse'
    loss_dict1 = compute_loss1( results , batch , epoch , config , class_range = (0,10) , is_training = is_training , mse_attribute = False )

    results['fc'] = results['s2'].clone()
    config.net['type'] = 'all'
    loss_dict2 = compute_loss1( results , batch , epoch , config , class_range , is_training = is_training , mse_attribute = False )

    s1_extend = results['s2'].clone()
    for i in range(s1_extend.shape[0]):
        for j in range(s1_extend.shape[1]):
            s1_extend[i,j] = results['s1'][i,train_dataset.superclass_label_list[j]]
    
    p1 = F.softmax( s1_extend / config.loss['T'] , dim = 1 )
    p2 = F.softmax( results['s2'] / config.loss['T']  , dim =  1)
    kl_div = nn.KLDivLoss(size_average = False)
    loss_dict = {}
    loss_dict[ 'kl' ] = kl_div( torch.log( p2 ) , p1.detach() )
    for k in loss_dict1:
        if k in ['softmax','aam']:
            loss_dict[k] = loss_dict1[k] + loss_dict2[k]
            #print(type(loss_dict[k]), type(loss_dict1[k] ) , type(loss_dict2[k]) )
        else:
            loss_dict[k+'_l1'] = loss_dict1[k]
            loss_dict[k+'_l2'] = loss_dict2[k]
    config.net['type'] = old_type
    return loss_dict

    
def compute_loss5( results , batch , epoch , config , class_range , is_training = False ):
    s, m, k = config.loss['s'] , config.loss['m'] , config.test['k']
    loss_dict = {}
    labels = batch['label']

    if is_training:
        sum_exp = torch.exp(results['fc']).sum(1)
        loss_dict['err'] = 1 - torch.eq( labels  , torch.max( results['fc'] , 1 )[1] ).float().mean()
        topk , top_idx = torch.topk( torch.exp(results['fc'] ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #tqdm.write( "{}".format(topk[0]), file=sys.stdout )
        if epoch < config.loss['arcloss_start_epoch'] :
            loss_dict['softmax'] = cross_entropy(  s*results['fc'] , labels )

        else:
            cos_theta = results['fc']
            cos_theta_yi = cos_theta[( torch.arange(cos_theta.shape[0] , dtype = torch.int64 ) , labels )  ]
            sin_theta_yi = ( 1 - cos_theta_yi**2 ) **0.5
            phai_theta_yi = cos_theta_yi * cos_m - sin_theta_yi * sin_m
            phai_theta_yi = torch.where( cos_theta_yi > border , phai_theta_yi , -2. - phai_theta_yi )#the loss curve correction
            phai_theta = cos_theta
            phai_theta[ ( torch.arange( cos_theta.shape[0] , dtype = torch.int64 ) , labels ) ] = phai_theta_yi
            loss_dict['aam'] = cross_entropy( s * phai_theta , labels )
    else:
        #k = config.test['k']
        #fc = results['fc']

        predicts = get_predict( results , config , latent_class_features , class_range , mode = 2 )
        loss_dict['err'] = 1 - torch.eq( predicts , labels ).float().mean() 
        loss_dict['predicts'] = predicts 

    loss_dict['mse_latent_feature'] = torch.mean( ( results['latent_semantic_feats'] - results['latent_visual_feats'] )**2 )
    #loss_dict['mse_latent_feature'] = mse( results['latent_semantic_feats'] , F.normalize( results['latent_visual_feats'].detach() ) )  + mse( results['latent_visual_feats'] , F.normalize( results['latent_semantic_feats'].detach() ) )
    return loss_dict

def compute_loss6( results , batch , epoch , config , class_range , is_training = False ):
    s, m, k = config.loss['s'] , config.loss['m'] , config.test['k']
    loss_dict = {}
    labels = batch['label']

    if is_training:
        sum_exp = torch.exp(results['fc']).sum(1)
        loss_dict['err'] = 1 - torch.eq( labels  , torch.max( results['fc'] , 1 )[1] ).float().mean()
        topk , top_idx = torch.topk( torch.exp(results['fc'] ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #tqdm.write( "{}".format(topk[0]), file=sys.stdout )
        if epoch < config.loss['arcloss_start_epoch'] :
            loss_dict['softmax'] = cross_entropy(  s*results['fc'] , labels )

        else:
            cos_theta = results['fc']
            cos_theta_yi = cos_theta[( torch.arange(cos_theta.shape[0] , dtype = torch.int64 ) , labels )  ]
            sin_theta_yi = ( 1 - cos_theta_yi**2 ) **0.5
            phai_theta_yi = cos_theta_yi * cos_m - sin_theta_yi * sin_m
            phai_theta_yi = torch.where( cos_theta_yi > border , phai_theta_yi , -2. - phai_theta_yi )#the loss curve correction
            phai_theta = cos_theta
            phai_theta[ ( torch.arange( cos_theta.shape[0] , dtype = torch.int64 ) , labels ) ] = phai_theta_yi
            loss_dict['aam'] = cross_entropy( s * phai_theta , labels )
        loss_dict['mse_latent_feature'] = torch.mean( ( results['latent_semantic_feats'][ batch['label'] ] - results['latent_visual_feats'] )**2 )
    else:
        #k = config.test['k']
        #fc = results['fc']

        predicts = get_predict( results , config , latent_class_features , class_range , mode = 2 )
        loss_dict['err'] = 1 - torch.eq( predicts , labels ).float().mean() 
        loss_dict['mse_latent_feature'] = torch.mean( ( torch.Tensor(latent_class_features).cuda()[batch['label'] ] - results['latent_visual_feats'] )**2 )

    #loss_dict['mse_latent_feature'] = mse( results['latent_semantic_feats'] , F.normalize( results['latent_visual_feats'].detach() ) )  + mse( results['latent_visual_feats'] , F.normalize( results['latent_semantic_feats'].detach() ) )
    return loss_dict

def compute_total_loss( loss_dict , config ):
    ret = 0
    if 'softmax' in loss_dict:
        ret += loss_dict['softmax']
    elif 'aam' in loss_dict:
        ret += loss_dict['aam']
    if 'mse_attribute' in loss_dict:
        ret += config.loss['weight_mse_attribute'] * loss_dict['mse_attribute']
    if 'mse_weight' in loss_dict:
        ret += loss_dict['mse_weight']
    if 'mse_latent_feature' in loss_dict:
        ret += config.loss['weight_latent_feature'] * loss_dict['mse_latent_feature']
    if 'kl' in loss_dict:
        ret += config.loss['weight_kl'] + loss_dict['kl']
    #ret += loss_dict['mse_feature_']
    return ret
    
def backward( loss , net , optimizer , config):
    optimizer.zero_grad()
    '''
    if 'softmax' in loss:
        loss['softmax'].backward(retain_graph=True)
    if 'aam' in loss:
        loss['aam'].backward(retain_graph=True)
    #set_requires_grad( net.attribute , True )
    if config.loss['weight_mse_attribute'] != 0 :
        (config.loss['weight_mse_attribute'] * loss['mse_attribute']).backward(retain_graph=False)
    '''
    tot_loss = compute_total_loss( loss , config )
    tot_loss.backward()
    optimizer.step()

def main(config):
    init(config)


    global train_dataset
    train_dataset = ZeroDataset(config.train['train_img_list'], config , is_training = True , has_filename = True)
    train_dataloader = torch.utils.data.DataLoader(  train_dataset , batch_size = config.train['batch_size']  , shuffle = True , drop_last = True , num_workers = 5 , pin_memory = True) 
    print( "num_classes : {}".format( train_dataset.num_classes ) )
    print( "num_attributes : {}".format( train_dataset.num_attributes ) )

    net_name = config.net['name']
    if 'arc' in net_name :
        net_type = config.net['type']
        if config.net['type'] == 'coarse':
            val_dataset_name = ['zero','non_zero','all']
        else:
            val_dataset_name = ['non_zero']
    elif 'gcn' in net_name :
        val_dataset_name = ['zero','non_zero']
    elif 'qfsl' in net_name :
        val_dataset_name = ['zero','non_zero','all']
    elif 'hse' in net_name:
        val_dataset_name = ['non_zero']
    elif 'tede' in net_name:
        val_dataset_name = ['zero','non_zero']
    elif 'gde' in net_name:
        val_dataset_name = ['zero','non_zero']

    val_dataloaders = {}
    for k in val_dataset_name:
        val_dataset = ZeroDataset(config.train['val_img_list'][k], config, is_training= False , has_filename = True)
        val_dataloaders[k] = torch.utils.data.DataLoader(  val_dataset , batch_size = config.train['val_batch_size']  , shuffle = False , drop_last = True , num_workers = 0 , pin_memory = False) 


    net_kwargs = deepcopy( config.net )
    net_name = net_kwargs.pop('name')
    try:
        net_kwargs.pop('type')
    except:
        pass
    if 'gcn' in net_name :
        A_hat =  get_graph( config , train_dataset.num_classes , train_dataset.labelno_to_labelname, train_dataset.labelname_to_realname )

        gcn_kwargs = deepcopy( config.net['gcn'] )
        gcn = GCN( A_hat , **gcn_kwargs )
        feature_net_kwargs = deepcopy( config.net['feature_net'] )
        feature_net_name = feature_net_kwargs.pop('name')
        load_path = feature_net_kwargs.pop('load_path')
        feature_net = eval(feature_net_name)( **feature_net_kwargs )
        load_model( feature_net , load_path , epoch = 'best_non_zero' ,  strict = True )
        net = ZeroShotGCN( feature_net , gcn )
        net.cuda()
        feature_net.cuda()
        set_requires_grad( feature_net , False )

        global label_weight
        label_weight = feature_net.classifier.linear.weight
    elif 'hse' in net_name:
        net_kwargs.pop('input_shape')
        net_kwargs.pop('num_attributes')
        net = eval(net_name)( **net_kwargs )
        net.cuda()
    elif 'gde' in net_name:
        A_hat =  get_graph( config , train_dataset.num_classes , train_dataset.labelno_to_labelname, train_dataset.labelname_to_realname )
        net_kwargs['semantic_gconv_kwargs']['A'] = A_hat
        net = eval(net_name)(**net_kwargs)
        net.cuda()

    else:
        net = eval(net_name)( **net_kwargs )
        config.net['name'] = net_name
        net.cuda()

    parameters = net.parameters()

    if 'ris' in net_name:
        compute_loss = compute_loss2
    elif 'gcn' in net_name:
        parameters = net.gcn.parameters()
        compute_loss = compute_loss3
    elif 'qfsl' in net_name:
        compute_loss = partial(compute_loss1 , mse_attribute = False )
    elif 'hse' in net_name:
        compute_loss = partial( compute_loss4 )
    elif 'tede' in net_name:
        compute_loss = compute_loss5
    elif 'gde' in net_name:
        compute_loss = compute_loss6
    else:
        compute_loss = compute_loss1




    tb = TensorBoardX(config = config , sub_dir = config.train['sub_dir'] , log_type = ['train' , 'val' , 'val_zero' , 'val_all' , 'val_non_zero' , 'net'] )
    tb.write_net(str(net),silent=False)
    assert config.train['optimizer'] in ['Adam' , 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam( parameters , lr = config.train['learning_rate']  ,  weight_decay = config.loss['weight_l2_reg']) 
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD( parameters , lr = config.train['learning_rate'] , weight_decay = config.loss['weight_l2_reg'] , momentum = config.train['momentum'] , nesterov = config.train['nesterov'] )


    last_epoch = -1 
    if config.train['resume'] is not None:
        last_epoch = load_model( net , config.train['resume'] , epoch = config.train['resume_epoch']  ) 

    if config.train['resume_optimizer'] is not None:
        _ = load_optimizer( optimizer , net  , config.train['resume_optimizer'] , epoch = config.train['resume_epoch'])
        assert last_epoch == _

    
    global cross_entropy,mse
    cross_entropy = nn.CrossEntropyLoss().cuda()
    mse = nn.MSELoss().cuda()

    
    #train_loss_epoch_list = []

    t_list = [time()]
    #convlstm_params = net.module.convlstm.parameters()
    #net_params = net.module.parameters()
    t = time()

    for k,v in net.named_parameters():
        if v.requires_grad and v.grad is not None:
            try:
                tb.add_histogram( k , v , 0 , 'net' )
                tb.add_histogram( k+'_grad' , v.grad , 0 , 'net' )
            except Exception as e:
                print( "{} is not finite".format(k)   )
                raise e

    def find_temp_best_lr( epoch , **kwargs ):
            
        def f(result,batch):
            return compute_loss( result['fc'] , batch['label'] , epoch = epoch , arcloss_start_epoch = config.loss['arcloss_start_epoch'] , s=  config.loss['s'] ,  is_training = True )
        forward_fn = lambda batch : net( batch['img'] , use_normalization = True )
        backward_fn = lambda loss , optimizer : backward( loss , net , optimizer , config )
        best_lr =  find_best_lr( f  , compute_total_loss , backward , net.module if isinstance(net,nn.DataParallel) else net ,  optimizer , iter( train_dataloader ) , forward_fn , **kwargs  )
        os.system('mkdir -p plot')
        plt.savefig('plot/epoch_{}_best_lr.jpg'.format(epoch))
        plt.close('all')
        return best_lr

    best_metric = {}
    for k in val_dataloaders:
        best_metric[k] = 1e9
    log_parser = LogParser()

    num_epochs_cycle = [int(config.train['cycle_mult'] ** i) * int(config.train['cycle_mult'] ** (config.train['num_cycles'] - i -1 ) )  for i in range(config.train['num_cycles'])   ] 
    cycle_borders = np.array( [0] + num_epochs_cycle ).cumsum()


    if config.train['last_epoch'] is not None:
        last_epoch = config.train['last_epoch']
    for epoch in tqdm(range( last_epoch + 1  , config.train['num_epochs'] ) , file = sys.stdout , desc = 'epoch' , leave=False ):
        #set_learning_rate( optimizer , epoch )

        if not config.train['mannual_learning_rate']:
            for idx , v in enumerate(cycle_borders):
                if epoch >= v:
                    cycle_idx =  idx
            if epoch == cycle_borders[cycle_idx] :
                #temp_lr = optimizer.param_groups[0].get('lr')
                #best_lr = find_temp_best_lr(epoch , start_lr = temp_lr if temp_lr is not None else 1e-5 ,num_iters = len(train_dataloader))
                best_lr = find_temp_best_lr(epoch , start_lr = 1e-5 ,num_iters = len(train_dataloader))
            cycle_len = int(config.train['cycle_mult'] ** cycle_idx)
            epoch_in_cycle = (epoch - cycle_borders[cycle_idx]) % cycle_len

           #tb.add_scalar( 'lr' , optimizer.param_groups[0]['lr'] , epoch*len(train_dataloader)  , 'train')
        log_dicts = {}

        #train
        def train():
            class_range = [0,config.train['num_all_classes'] - config.train['num_test_classes'] - config.train['num_zero_classes']]
            log_t = time()
            train_loss_log_list = [] 
            net.train()
            if 'gcn' in  net_name:
                data_loader = zip( range(len(train_dataset)//config.train['batch_size']) , range(len(train_dataset) // config.train['batch_size']) )
                length = len(train_dataset) // config.train['batch_size']
            else:
                data_loader = train_dataloader
                length = len(train_dataloader)
            for step , batch in tqdm(enumerate( data_loader) , total = length , file = sys.stdout , desc = 'training' , leave=False):
                if config.train['mannual_learning_rate']:
                    mannual_learning_rate(optimizer,epoch,step,length,config)
                assert len(optimizer.param_groups) == 1 
                if not config.train['mannual_learning_rate']:
                    for param_group in optimizer.param_groups:
                        num_iters_a_cycle = len(train_dataloader) * cycle_len
                        iter_in_cycle = epoch_in_cycle * len(train_dataloader) + step
                        param_group['lr'] = best_lr * np.cos( np.pi / 2 / num_iters_a_cycle * iter_in_cycle )
                tb.add_scalar( 'lr' , optimizer.param_groups[0]['lr'] , epoch*len(train_dataloader) + step , 'train')
                if not 'gcn' in net_name:
                    for k in batch:
                        if not k in ['filename']:
                            batch[k] = batch[k].cuda(async =  True) 
                            batch[k].requires_grad = False
                #sanity check
                '''
                label = batch['label'][0]
                labelname = train_dataset.labelno_to_labelname[int(label.cpu().detach().numpy())]
                a = open('./data/DatasetA_train_20180813/attributes_per_class_cleaned.txt').read().strip().split('\n')
                labelname_to_attribute = {}
                for line in a:
                    line = line.split('\t')
                    labelname_to_attribute[line[0]] = np.array(line[1:])
                assert batch['attribute'][0].cpu().detach().numpy() == labelname_to_attribute[labelname] , "{}\n{}".format( batch['attribute'][0].cpu().detach().numpy() , labelname_to_attribute[labelname] )
                '''

                #results = net( batch['img'] , use_normalization = True if epoch >= config.loss['arcloss_start_epoch'] else False)
                if 'gcn' in net_name:
                    results = net( None , torch.Tensor( train_dataset.class_attributes ).cuda(),  use_normalization = True )
                elif 'tede' in net_name:
                    results = net( batch['img']  , batch['attribute'] )
                elif 'gde' in net_name:
                    results = net( batch['img'] , torch.Tensor( train_dataset.class_attributes ).cuda() )
                else:
                    #results = net( batch['img'] , torch.Tensor( train_dataset.class_attributes[:train_num_classes] ).cuda(),  use_normalization = True )
                    results = net( batch['img'] , torch.Tensor( train_dataset.class_attributes[:train_num_classes] ).cuda(),  use_normalization = True )

                loss_dict = compute_loss( results , batch  , epoch , config , class_range ,  is_training= True )
                backward( loss_dict , net , optimizer , config )

                for k in loss_dict:
                    if len(loss_dict[k].shape) == 0 :
                        loss_dict[k] = float(loss_dict[k].cpu().detach().numpy())
                        tb.add_scalar( k , loss_dict[k] , epoch*len(train_dataloader) + step , 'train' )
                    else:
                        loss_dict[k] = loss_dict[k].cpu().detach().numpy()
                train_loss_log_list.append( { k:loss_dict[k] for k in loss_dict} )

                if step % config.train['log_step'] == 0 and epoch == last_epoch + 1 :
                    for k in filter( lambda x:isinstance(loss_dict[x],float) , loss_dict):
                        tqdm.write( "{} : {} ".format(k,loss_dict[k] )  , file=sys.stdout )
                            
                    for k,v in net.named_parameters():
                        if v.requires_grad and v.grad is not None:
                            try:
                                tb.add_histogram( k , v , (epoch)*len(train_dataloader) + step , 'net' )
                            except Exception as e:
                                print( "{} is not finite".format(k)   )
                                raise e
                            try:
                                tb.add_histogram( k+'_grad' , v.grad , (epoch)*len(train_dataloader) + step , 'net' )
                            except Exception as e:
                                print( "{}.grad is not finite".format(k)   )
                                raise e


            log_dict = {}
            for k in train_loss_log_list[0]:
                if isinstance(train_loss_log_list[0][k] , float ):
                    log_dict[k] = float( np.mean( [dic[k] for dic in train_loss_log_list ]  ) )
                else:
                    log_dict[k] = np.concatenate(  [dic[k] for dic in train_loss_log_list ] , axis = 0  )
            #log_dict = { k: float( np.mean( [ dic[k] for dic in train_loss_log_list ] )) for k in train_loss_log_list[0] }
            return log_dict

        log_dicts['train'] = train() 

        #validate
        net.eval()
        def validate( key , val_dataloader):
            if key == 'non_zero':
                class_range = (0, config.train['num_all_classes'] - config.train['num_test_classes'] - config.train['num_zero_classes'])
            elif key == 'zero':
                #class_range = ( config.train['num_all_classes'] - config.train['num_test_classes'] - config.train['num_zero_classes']  , config.train['num_all_classes'] - config.train['num_test_classes'])
                class_range = ( config.train['num_all_classes'] - config.train['num_test_classes'] - config.train['num_zero_classes']  , config.train['num_all_classes'] ) 
            elif key == 'all':
                class_range = (0,config.train['num_all_classes'] )


            if 'type' in config.net and config.net['type'] == 'coarse':
                class_range = (0,100)

            if 'tede'  in net_name or 'gde' in net_name:
                global latent_class_features
                latent_class_features = net.forward_attribtues( torch.Tensor( train_dataset.class_attributes ).cuda() )['latent_semantic_feats'].cpu().detach().numpy()

            val_loss_log_list= [ ]
            with torch.no_grad():
                first_val = False
                for step , batch in tqdm( enumerate( val_dataloader ) , total = len( val_dataloader ) , desc = 'validating' , leave = False  ):
                    for k in batch:
                        if not k in ['filename']:
                            batch[k] = batch[k].cuda(async =  True) 
                            batch[k].requires_grad = False

                    if 'gcn' in net_name:
                        results = net( batch['img'] , torch.Tensor( train_dataset.class_attributes ).cuda(),  use_normalization = True )
                    elif 'tede' in net_name:
                        results = net( batch['img']  , batch['attribute'] )
                    elif 'gde' in net_name:
                        results = net( batch['img'] )
                    else:
                        results = net( batch['img'] , torch.Tensor( train_dataset.class_attributes ).cuda(),  use_normalization = True )

                    loss_dict = compute_loss( results , batch , epoch , config , class_range = class_range , is_training = False )


                    for k in loss_dict:
                        if len(loss_dict[k].shape) == 0 :
                            loss_dict[k] = float(loss_dict[k].cpu().detach().numpy())
                        else:
                            loss_dict[k] = loss_dict[k].cpu().detach().numpy()
                    val_loss_log_list.append( { k:loss_dict[k] for k in loss_dict} )
                #log_dict = { k: float( np.mean( [ dic[k] for dic in val_loss_log_list ] )) for k in val_loss_log_list[0] }
                log_dict = {}
                for k in val_loss_log_list[0]:
                    if isinstance(val_loss_log_list[0][k] , float ) :
                        log_dict[k] = float( np.mean( [dic[k] for dic in val_loss_log_list ]  ) )
                    else:
                        log_dict[k] = np.concatenate(  [dic[k] for dic in val_loss_log_list ] , axis = 0  )
                return log_dict

        for k in val_dataloaders:
            log_dicts['val_'+k] =  validate( k ,  val_dataloaders[k]  ) 

        #save
        for k in val_dataloaders:
            if best_metric[k] > log_dicts['val_'+k][config.train['save_metric']]:
                save_model( net , tb.path , 'best_{}'.format(k)  )
                save_optimizer( optimizer , net , tb.path , 'best_{}'.format(k)  )
                best_metric[k] = log_dicts['val_'+k][config.train['save_metric']]
            if epoch == config.train['num_epochs'] - 1 :
                save_model( net , tb.path , 'last_{}'.format(k)  )
                save_optimizer( optimizer , net , tb.path , 'last_{}'.format(k)  )

        #log
        for k,v in net.named_parameters():
            if v.requires_grad and v.grad is not None:
                try:
                    tb.add_histogram( k , v , (epoch+1)*len(train_dataloader) , 'net' )
                except Exception as e:
                    print( "{} is not finite".format(k)   )
                    raise e
                try:
                    tb.add_histogram( k+'_grad' , v.grad , (epoch+1)*len(train_dataloader) , 'net' )
                except Exception as e:
                    print( "{}.grad is not finite".format(k)   )
                    raise e

        for tag in log_dicts:
            if 'val' in tag:
                for k,v in log_dicts[tag].items():
                    if isinstance( v  , float ) :
                        tb.add_scalar( k , v , (epoch+1)*len(train_dataloader) , tag ) 
                    else:
                        tb.add_histogram( k , v , (epoch+1)*len(train_dataloader) , tag )

        if 'gcn' in net_name:
            num_imgs = len(train_dataset) // config.train['batch_size']
        else:
            num_imgs = config.train['batch_size'] * len(train_dataloader) + config.train['val_batch_size'] *sum( [len(val_dataloaders[k]) for k in val_dataloaders ] )
        log_msg = log_parser.parse_log_dict( log_dicts , epoch , optimizer.param_groups[0]['lr'] , num_imgs , config = config )
        tb.write_log(  log_msg  , use_tqdm = True )

    for k in best_metric:
        tb.write_log("{} : {}".format( k ,best_metric[k]) )
    return { 'log_path':tb.path ,**best_metric }


if __name__ == '__main__':
    import train_config as config
    main(config)
