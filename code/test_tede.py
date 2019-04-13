import torch
import argparse
from network import *
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model
from time import time
import os
import importlib
import train_config as config
import datetime



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume')
    parser.add_argument('-input_list')
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--output_list',default='../submit/tede_{}.txt'.format( datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
 ))
    parser.add_argument('--resume_epoch',default=None)
    parser.add_argument('--has_label',dest='has_label',action = 'store_true')
    parser.set_defaults( has_label = False )

    return parser.parse_args()

def init(config):
    global zero_class , class_attributes , superclass_idx , labelname_to_labelno , labelno_to_labelname , superclass_zero_idx , superclass_nonzero_idx , test_class
    zero_class = open( config.train['zero_class_file'] ).read().strip().split('\n')
    nonzero_class = open( config.train['nonzero_class_file'] ).read().strip().split('\n')
    test_class = open( config.train['test_class_file'] ).read().strip().split()
    class_attributes = np.loadtxt( config.train['class_attributes'] )
    label_dict_list = open(config.train['label_dict_file']).read().strip().split('\n')
    labelname_to_labelno = {}
    labelno_to_labelname = {}
    for line in label_dict_list:
        labelname_to_labelno[line.split('\t')[0]] = int(line.split('\t')[1])
        labelno_to_labelname[int(line.split('\t')[1])] = line.split('\t')[0]
    superclass_idx = [[] for i in range(10)]
    superclass_zero_idx = [[] for i in range(10)]
    superclass_nonzero_idx = [[] for i in range(10)]
    for idx,a in enumerate(class_attributes):
        superclass_idx[get_superclass(a)].append(idx)
        if labelno_to_labelname[idx] not in nonzero_class:#in zero_class or in test_class
            superclass_zero_idx[get_superclass(a)].append(idx)
        else:
            superclass_nonzero_idx[get_superclass(a)].append(idx)

    super_class_name = ['animal','transportation','clothes','plant','tableware','others','device','building','food','scene']
    print('zero_class:')
    for idx,idx_list in enumerate(superclass_zero_idx):
        print( "{} {} ".format(super_class_name[idx] ,  len(idx_list)), end = ""  )
    print('')
    print('nonzero_class:')
    for idx,idx_list in enumerate(superclass_nonzero_idx):
        print( "{} {} ".format(super_class_name[idx] ,  len(idx_list)), end = ""  )
    print('')

def main(args,config):
    init(config)

    
    train_dataset = ZeroDataset( config.train['train_img_list'] , config, is_training = False , has_filename = True , has_label = args.has_label)
    dataset = ZeroDataset( args.input_list , config , is_training = False , has_filename = True , has_label = args.has_label)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )
            
    net_kwargs = deepcopy( config.net )


    net_name = net_kwargs.pop('name')
    try:
        net_kwargs.pop('type')
    except:
        pass
    net = eval(net_name)(  **net_kwargs )
    config.net['name'] = net_name
    net.cuda()
    #print(net)


    last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  , strict = True) 
    

    feats_list = []
    output_list = []
    fc_list = []
    predict_list = []
    tt = time()
    net.eval()
    global latent_class_features
    latent_class_features = net.forward_attribtues( torch.Tensor( train_dataset.class_attributes ).cuda() )['latent_semantic_feats'].cpu().detach().numpy()
    with torch.no_grad():
        for step , batch in tqdm(enumerate( dataloader ) , total = len(dataloader) ):
            for k in batch:
                if 'img' in k or 'attribute' in k :
                    batch[k] = batch[k].cuda(async = True)
                    batch[k].requires_grad = False

            results = net( batch['img'] ) 
            if args.has_label:
                #class_range = ( config.train['num_all_classes'] - config.train['num_test_classes'] - config.train['num_zero_classes'] ,  config.train['num_all_classes'] - config.train['num_test_classes'] )
                class_range = ( config.train['num_all_classes'] - config.train['num_test_classes'] - config.train['num_zero_classes'] ,  config.train['num_all_classes'] )
            else:
                class_range = (config.train['num_all_classes'] - config.train['num_test_classes'] , config.train['num_all_classes'] )
                #class_range = ( 0 , config.train['num_all_classes'] )
            predicts = get_predict( results , config , latent_class_features , class_range = class_range ,  mode = 2  )
            predict_list.append( predicts.cpu().detach().numpy() )
            fc = torch.matmul( F.normalize( results['latent_visual_feats'] )  , F.normalize( torch.Tensor( latent_class_features ).cuda() ).transpose(0,1) ) 
            fc_list.append( fc.cpu().detach().numpy() )
            for filename , predict in  zip( batch['filename'] , predicts.cpu().detach().numpy()  ):
                line = filename.split('/')[-1] + '\t' +  train_dataset.labelno_to_labelname[ predict ]
                output_list.append( line )
    #feats = np.vstack( feats_list )
    with open( args.output_list , 'w' ) as fp:
        fp.write( '\n'.join( output_list ) + '\n' )
    if args.has_label:
        cmd = 'python compare.py {} {}'.format(args.input_list , args.output_list) 
        os.system(cmd)
        os.system( 'rm {}'.format(args.output_list) )

    #np.savetxt(args.output_list, feats, fmt='%.18e', delimiter=',')
    #os.system('python /home/wtw/scripts/test_lfw.py {} {} {}'.format(args.output_list,'/home/hzl/dataset/lfw-list.txt','/home/hzl/dataset/lfw-pairs.txt'))


    fc = np.concatenate( fc_list , axis = 0 )
    y = np.concatenate( predict_list , axis = 0  )
    fc_list = []
    for single_fc in fc:
        #fc_list.append( {train_dataset.labelno_to_labelname[ labelno ] :v for labelno,v in enumerate(single_fc)} )
        fc_list.append( {train_dataset.labelno_to_labelname[ labelno ] :v for labelno,v in enumerate(single_fc)} )
    cnt = 0
    cnt_false = 0
    new_fc_list = []
    for fcc , yy in zip( fc_list , y ):
        fccc = { k:fcc[k] for k in (test_class ) }
        #fccc = fcc

        new_fc_list.append(fccc)
        #assert train_dataset.labelno_to_labelname[int(yy.detach().cpu().numpy())] in zero_class + test_class
        assert train_dataset.labelno_to_labelname[int(yy)] in  test_class

        if  max( fccc  , key = fccc.get ) != train_dataset.labelno_to_labelname[int(yy)] :
            #print( fcc[max( fccc  , key = fccc.get )] , fcc[ train_dataset.labelno_to_labelname[int(yy.detach().cpu().numpy())] ] )
 #           print(fccc)
            cnt_false += 1
        cnt += 1
    print( cnt_false ,  cnt )
    return {'fc': new_fc_list , 'y':y}

if __name__ == '__main__':
    import train_config as config
    args = parse_args()
    main(args,config)
