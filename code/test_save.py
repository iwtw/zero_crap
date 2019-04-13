import torch
import argparse
from network import arc_resnet18
from custom_utils import get_predict
from dataset import ZeroDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model,set_requires_grad
from time import time
import os
import train_config as config
from copy import deepcopy



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume')
    parser.add_argument('-input_list')
    parser.add_argument('--output_list',type=str,default='./output.list')
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--resume_epoch',default=None)
    parser.add_argument('--has_label',dest='has_label',action='store_true')
    parser.set_defaults(has_label=False)
    return parser.parse_args()


if __name__ == '__main__' :
    args = parse_args()

    dataset = ZeroDataset(args.input_list,config, is_training = False , has_label=args.has_label,has_filename=True)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )
    
    net_kwargs = deepcopy( config.net )
    try:
        net_kwargs.pop('type')
    except:
        pass
    net_name = net_kwargs.pop('name')
    net = eval(net_name)( **net_kwargs )
    net.cuda()
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
            y = get_predict( results , config , dataset.class_attributes )

            input_filename_list = batch['filename']
            for predict,filename in zip(y,batch['filename']):
                output_list.append( '{}\t{}'.format(filename.split('/')[-1] , dataset.labelno_to_labelname[int(predict.cpu().detach().numpy())])  )
    
    with open(args.output_list,'w') as fp:
        fp.write( '\n'.join( output_list ) + '\n' )


    if args.has_label:
        cmd = 'python compare.py {} {} '.format( args.output_list , args.input_list )
        os.system( cmd )
        os.system( 'rm {}'.format( args.output_list ) )
