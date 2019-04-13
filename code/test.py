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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('resume')
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--output_list',default='./temp/output.list')
    parser.add_argument('--resume_epoch',default=None)

    return parser.parse_args()

#from stackoverflow user "m.kocikowski"
def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

if __name__ == '__main__' :
    args = parse_args()
    assert args.dataset in ['lfw','cfp']
    if args.dataset == 'lfw':
        input_list = '/home/hzl/dataset/lfw-list.txt'

    print(input_list)

    train_dataset = MSCelebDataset(config.train['train_img_list'],is_training=False,has_label=True)
    dataset = MSCelebDataset(input_list,is_training=False,has_label=False)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = True )
    
            
    net_name = config.net.pop('name')
    net = eval(net_name)( num_classes = train_dataset.num_classes , **config.net )
    del( train_dataset )
    net.features = nn.DataParallel( net.features )
    config.net['name'] = net_name
    net.cuda()
    print(net)


    last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  , strict = False) 
    
    log_dir = '{}/test/{}'.format(args.resume,args.dataset)
    os.system('mkdir -p {}'.format(log_dir) )

    feats_list = []
    tt = time()
    net.eval()
    with torch.no_grad():
        for step , batch in tqdm(enumerate( dataloader ) , total = len(dataloader) ):
            for k in batch:
                batch[k] = batch[k].cuda(async = True)
                batch[k].requires_grad = False

            results = net( batch['img'] , use_normalization = True )
            feats_list.append( results['feature'].detach().cpu().numpy() )
    feats = np.vstack( feats_list )
    np.savetxt(args.output_list, feats, fmt='%.18e', delimiter=',')
    os.system('python /home/wtw/scripts/test_lfw.py {} {} {}'.format(args.output_list,'/home/hzl/dataset/lfw-list.txt','/home/hzl/dataset/lfw-pairs.txt'))


    #with open('{}/psnr.txt'.format(log_dir),'a') as log_fp:
    #    log_fp.write( 'epoch {} : psnr {}'.format( last_epoch , psnr ) )
