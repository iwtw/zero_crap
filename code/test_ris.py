import torch
import argparse
from network import *
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
    parser.add_argument('--resume_try_no',type=int,default=0)
    parser.add_argument('--num_attributes',type=int,default=26)
    parser.add_argument('--has_label',dest='has_label',action='store_true')
    parser.set_defaults(has_label=False)
    parser.set_defaults(attribute=False)
    return parser.parse_args()


if __name__ == '__main__' :
    args = parse_args()

    dataset = ZeroDataset(args.input_list,config.train['attribute_file'] , config.train['label_dict_file'] , config.train['attribute_index'] , is_training = False , has_label=args.has_label,has_filename=True)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = False )
    
    output_list = []
    attributes = np.zeros( (len(dataset) , args.num_attributes) , dtype=np.float32 )
    filenames = []
    for i in range(args.num_attributes):
        net_name = config.net.pop('name')
        net = eval(net_name)( **config.net )
        net.cuda()
        config.net['name'] = net_name
        last_epoch = load_model( net , args.resume +'/attribute{}/{}'.format(i,args.resume_try_no) , epoch = args.resume_epoch  ) 
        

        tt = time()
        net.eval()
        cnt = 0
        with torch.no_grad():
            for step , batch in tqdm(enumerate( dataloader ) , total = len(dataloader) ):
                for k in batch:
                    if 'img' in k:
                        batch[k] = batch[k].cuda(async = True)
                        batch[k].requires_grad = False

                
                results = net( batch['img'] , None , True)
                attrs = results['attribute']
                for attr in attrs:
                    attributes[cnt,i] = attr.cpu().detach().numpy()
                    cnt += 1
                if i == 0 :
                    for filename in batch['filename']:
                        filenames.append(filename)
    attr_list = dataset.attr_list[:,:args.num_attributes]
    for filename,attr in zip(filenames,attributes):
        y = get_predict( {'attribute':torch.Tensor(attr.reshape(1,-1)).cuda()} , config , attr_list , mode = 0 )
        output_list.append('{}\t{}'.format(filename.split('/')[-1] , dataset.labelno_to_labelname[int(y.cpu().detach().numpy())]))


    with open(args.output_list,'w') as fp:
        fp.write( '\n'.join( output_list ) + '\n' )
