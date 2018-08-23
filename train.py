import torch
import torch.nn as nn
import numpy as np
from log import TensorBoardX,LogParser
from utils import *
from dataset import *
from tqdm import tqdm
from time import time
from network import arc_resnet18,get_predict
import sys
import pynvml
from functools import partial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn


def init(config):
    m = config.loss['m']
    global cos_m , sin_m , border , attribute_zero 
    cos_m = np.cos( m )
    sin_m = np.sin( m )
    border = np.cos( math.pi - m )
    attribute_list = open(config.train['attribute_file']).read().strip().split('\n')
    attribute_zero_list = []
    for i , line in enumerate( attribute_list):
        line = line.split('\t')
        if 191 <= int(line[0][3:]) <= 240 :
            attribute_zero_list.append(line[1:])
            assert len(attribute_zero_list[-1]) == 26
    attribute_zero = np.array( attribute_zero_list , np.float32)
    assert len(attribute_zero) == 50



def mannual_learning_rate( optimizer , epoch , lrs , bounds ):
    for idx in range(len(bounds) - 1):
        if bounds[idx] <= epoch and epoch < bounds[idx+1]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrs[idx]

def compute_loss( results , batch , epoch , config ,  is_training = False  ):
    s, m, k = config.loss['s'] , config.loss['m'] , config.test['k']
    loss_dict = {}
    #print(results['fc'].shape ,  batch['label'].max() )
    if is_training:
        sum_exp = torch.exp(results['fc']).sum(1)
        loss_dict['acc'] = torch.eq( batch['label']  , torch.max( results['fc'] , 1 )[1] ).float().mean()
        topk , top_idx = torch.topk( torch.exp(results['fc'] ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #tqdm.write( "{}".format(topk[0]), file=sys.stdout )
        if epoch < config.loss['arcloss_start_epoch'] :
            loss_dict['softmax'] = cross_entropy(  s*results['fc'] , batch['label'] )

        else:
            cos_theta = results['fc']
            cos_theta_yi = cos_theta[( torch.arange(cos_theta.shape[0] , dtype = torch.int64 ) , batch['label'] )  ]
            sin_theta_yi = ( 1 - cos_theta_yi**2 ) **0.5
            phai_theta_yi = cos_theta_yi * cos_m - sin_theta_yi * sin_m
            phai_theta_yi = torch.where( cos_theta_yi > border , phai_theta_yi , -2. - phai_theta_yi )#the loss curve correction
            phai_theta = cos_theta
            phai_theta[ ( torch.arange( cos_theta.shape[0] , dtype = torch.int64 ) , batch['label'] ) ] = phai_theta_yi
            loss_dict['aam'] = cross_entropy( s * phai_theta , batch['label'] )
    else:
        #for i in range(k):
        #    loss_dict['top{}'.format(i+1)] = torch.mean( topk[:,i] )

        
        #top_idx = top_idx.cpu().detach().numpy()

        predicts = get_predict( results['fc'] , config , attr_list )
        loss_dict['acc'] =  torch.eq( predicts , batch['label'] ).float().mean() 

    loss_dict['mse_attribute'] = mse( results['attribute'] , batch['attribute'] )
        #loss_dict['mse_feature_'] = mse( results['feature_'] , results['feature'].detach() )
    return loss_dict

def compute_total_loss( loss_dict ):
    ret = 0
    if 'softmax' in loss_dict:
        ret += loss_dict['softmax']
    elif 'aam' in loss_dict:
        ret += loss_dict['aam']
    ret += loss_dict['mse_attribute']
    #ret += loss_dict['mse_feature_']
    self.has_filename = has_filename
    return ret
    
    


def backward( loss , net , optimizer ):
    optimizer.zero_grad()
    '''
    set_requires_grad( net.attribute , False )
    set_requires_grad( net.inverse_attribute , False )
    '''
    if 'softmax' in loss:
        loss['softmax'].backward(retain_graph=True)
    if 'aam' in loss:
        loss['aam'].backward(retain_graph=True)
    #set_requires_grad( net.attribute , True )
    loss['mse_attribute'].backward(retain_graph=False)
    '''
    set_requires_grad( net.attribute , False )
    set_requires_grad( net.inverse_attribute , True )
    loss['mse_feature_'].backward(  )
    set_requires_grad( net.attribute , True )
    '''
    optimizer.step()

def main(config):
    init(config)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)


    train_dataset = ZeroDataset(config.train['train_img_list'],config.train['attribute_file'],config.train['label_dict_file'],is_training = True )
    train_dataloader = torch.utils.data.DataLoader(  train_dataset , batch_size = config.train['batch_size']  , shuffle = True , drop_last = False , num_workers = 5 , pin_memory = True) 
    global attr_list
    attr_list = train_dataset.attr_list
    print( "num_classes : {}".format( train_dataset.num_classes ) )
    print( "num_attributes : {}".format( train_dataset.num_attributes ) )

    val_dataset_name = ['zero','multi','all']
    val_dataloaders = {}
    for k in val_dataset_name:
        val_dataset = ZeroDataset(config.train['val_img_list'][k] ,  config.train['attribute_file'] , config.train['label_dict_file'] , is_training= False)
        val_dataloaders[k] = torch.utils.data.DataLoader(  val_dataset , batch_size = config.train['val_batch_size']  , shuffle = False , drop_last = False , num_workers = 0 , pin_memory = False) 


    net_name = config.net.pop('name')
    net = eval(net_name)( **config.net )
   # net = nn.DataParallel( net )
    config.net['name'] = net_name
    #net.features = nn.DataParallel( net.features )
    net.cuda()
    tb = TensorBoardX(config = config , sub_dir = config.train['sub_dir'] , log_type = ['train' , 'val' , 'val_zero' , 'val_all' , 'val_multi'] )
    tb.write_net(str(net))
    assert config.train['optimizer'] in ['Adam' , 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam( net.parameters() , lr = config.train['learning_rate']  ,  weight_decay = config.loss['weight_l2_reg']) 
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD( net.parameters() , lr = config.train['learning_rate'] , weight_decay = config.loss['weight_l2_reg'] , momentum = config.train['momentum'] , nesterov = config.train['nesterov'] )


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

    def find_temp_best_lr( epoch , **kwargs ):
            
        def f(result,batch):
            return compute_loss( result['fc'] , batch['label'] , epoch = epoch , arcloss_start_epoch = config.loss['arcloss_start_epoch'] , s=  config.loss['s'] ,  is_training = True )
        forward_fn = lambda batch : net( batch['img'] , use_normalization = True )
        backward_fn = lambda loss , optimizer : backward( loss , net , optimizer )
        best_lr =  find_best_lr( f  , compute_total_loss , backward , net.module if isinstance(net,nn.DataParallel) else net ,  optimizer , iter( train_dataloader ) , forward_fn , **kwargs  )
        os.system('mkdir -p plot')
        plt.savefig('plot/epoch_{}_best_lr.jpg'.format(epoch))
        plt.close('all')
        return best_lr

    best_acc = {}
    for k in val_dataloaders:
        best_acc[k] = 0
    log_parser = LogParser()

    num_epochs_cycle = [int(config.train['cycle_mult'] ** i) * int(config.train['cycle_mult'] ** (config.train['num_cycles'] - i -1 ) )  for i in range(config.train['num_cycles'])   ] 
    cycle_borders = np.array( [0] + num_epochs_cycle ).cumsum()
    for epoch in tqdm(range( last_epoch + 1  , config.train['num_epochs'] ) , file = sys.stdout , desc = 'epoch' , leave=False ):
        #set_learning_rate( optimizer , epoch )

        if config.train['mannual_learning_rate']:
            mannual_learning_rate(optimizer,epoch,config.train['lrs'],config.train['lr_bounds'])
        else:
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
            log_t = time()
            train_loss_log_list = [] 
            net.train()
            for step , batch in tqdm(enumerate( train_dataloader) , total = len(train_dataloader) , file = sys.stdout , desc = 'training' , leave=False):
                assert len(optimizer.param_groups) == 1 
                if not config.train['mannual_learning_rate']:
                    for param_group in optimizer.param_groups:
                        num_iters_a_cycle = len(train_dataloader) * cycle_len
                        iter_in_cycle = epoch_in_cycle * len(train_dataloader) + step
                        param_group['lr'] = best_lr * np.cos( np.pi / 2 / num_iters_a_cycle * iter_in_cycle )
                tb.add_scalar( 'lr' , optimizer.param_groups[0]['lr'] , epoch*len(train_dataloader) + step , 'train')
                for k in batch:
                    batch[k] = batch[k].cuda(async =  True) 
                    batch[k].requires_grad = False

                #results = net( batch['img'] , use_normalization = True if epoch >= config.loss['arcloss_start_epoch'] else False)
                attribute_zero_tensor = torch.Tensor( attribute_zero ).cuda()
                attribute_zero_tensor.requires_grad = False
                results = net( batch['img'] , attribute_zero_tensor ,  use_normalization = True )

                loss_dict = compute_loss( results , batch  , epoch , config ,  is_training= True )
                backward( loss_dict , net , optimizer )

                for k in loss_dict:
                    loss_dict[k] = float(loss_dict[k].cpu().detach().numpy())
                train_loss_log_list.append( { k:loss_dict[k] for k in loss_dict} )
                for k,v in loss_dict.items():
                    tb.add_scalar( k , v , epoch*len(train_dataloader) + step , 'train' )
                if step % config.train['log_step'] == config.train['log_step'] - 1  and epoch == last_epoch + 1 :
                    temp_t = time()
                    train_loss_log_dict = { k: float( np.mean( [ dic[k] for dic in train_loss_log_list[-config.train['log_step']:]] )) for k in train_loss_log_list[0] }
                    log_msg = 'step {} , {} imgs/s'.format(step,config.train['batch_size'] * config.train['log_step'] / (temp_t - log_t) )
                    log_msg += " | train : "
                    for idx,k_v in enumerate(train_loss_log_dict.items()):
                        k,v = k_v
                        if k == 'acc':
                            log_msg += "{} {:.3%} {} ".format(k,v,',')
                        else:
                            log_msg += "{} {:.5f} {} ".format(k,v,',' if idx < len(train_loss_log_dict) - 1 else '')
                    tb.write_log( log_msg , use_tqdm = True)
                    log_t = temp_t
            log_dict = { k: float( np.mean( [ dic[k] for dic in train_loss_log_list ] )) for k in train_loss_log_list[0] }
            return log_dict
        log_dicts['train'] = train() 

        #validate
        def validate(val_dataloader):
            val_loss_log_list= [ ]
            net.eval()
            with torch.no_grad():
                first_val = False
                for step , batch in tqdm( enumerate( val_dataloader ) , total = len( val_dataloader ) , desc = 'validating' , leave = False  ):
                    for k in batch:
                        batch[k] = batch[k].cuda(async =  True) 
                        batch[k].requires_grad = False
                    attribute_zero_tensor = torch.Tensor( attribute_zero ).cuda()
                    attribute_zero_tensor.requires_grad = False

                    results = net( batch['img'] , attribute_zero_tensor ,  use_normalization = True )

                    loss_dict = compute_loss( results , batch , epoch , config , is_training = False )


                    for k in loss_dict:
                        loss_dict[k] = float(loss_dict[k].cpu().detach().numpy())
                    val_loss_log_list.append( { k:loss_dict[k] for k in loss_dict} )
                log_dict = { k: float( np.mean( [ dic[k] for dic in val_loss_log_list ] )) for k in val_loss_log_list[0] }
                return log_dict
        for k in val_dataloaders:
            log_dicts['val_'+k] =  validate( val_dataloaders[k]  ) 

        #save
        for k in val_dataloaders:
            if best_acc[k] < log_dicts['val_'+k]['acc']:
                save_model( net , tb.path , 'best_{}'.format(k)  )
                save_optimizer( optimizer , net , tb.path , 'best_{}'.format(k)  )
                best_acc[k] = log_dicts['val_'+k]['acc']

        #log
        for tag in log_dicts:
            if 'val' in tag:
                for k,v in log_dicts[tag].items():
                    tb.add_scalar( k , v , (epoch+1)*len(train_dataloader) , tag ) 

        log_msg = log_parser.parse_log_dict( log_dicts , epoch , optimizer.param_groups[0]['lr'] , config.train['batch_size'] * len(train_dataloader) + config.train['val_batch_size'] *sum( [len(val_dataloaders[k]) for k in val_dataloaders ] ) , config = config )
        tb.write_log(  log_msg  , use_tqdm = True )


if __name__ == '__main__':
    import train_config as config
    main(config)
