import torch
import argparse
from network import arc_resnet18,get_predict
from dataset import ZeroDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model,set_requires_grad
from time import time
import os
import train_config as config



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume')
    parser.add_argument('-input_list')
    parser.add_argument('--output_list',type=str,default='./output.list')
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--resume_epoch',default=None)
    return parser.parse_args()


if __name__ == '__main__' :
    args = parse_args()

    dataset = ZeroDataset(args.input_list,config.train['attribute_file'] , is_training = False , has_label=False,has_filename=True)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = False )
    
    net_name = config.net.pop('name')
    net = eval(net_name)( **config.net )
    net.cuda()
    config.net['name'] = net_name
    last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  ) 
    
    psnr_list = []
    output_list = []

    tt = time()
    net.eval()
    with torch.no_grad():
        for step , batch in tqdm(enumerate( dataloader ) , total = len(dataloader) ):
            for k in batch:
                if 'img' in k:
                    batch[k] = batch[k].cuda(async = True)
                    batch[k].requires_grad = False

            
            results = net( batch['img'] , None , True)
            y = get_predict( results['fc'] , config , dataset.attr_list )

            input_filename_list = batch['filename']
            for predict,filename in zip(y,batch['filename']):
                output_list.append( '{}\t{}'.format(filename.split('/')[-1] , dataset.label_idx_to_label_list[predict])  )
    
    with open(args.output_list,'w') as fp:
        fp.write( '\n'.join( output_list ) + '\n' )
