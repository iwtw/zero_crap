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
from train import compute_loss1
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume')
    parser.add_argument('--output_list',type=str,default='./output.list')
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--resume_epoch',default=None)
    parser.add_argument('--has_label',dest='has_label',action='store_true')
    parser.set_defaults(has_label=False)
    return parser.parse_args()


if __name__ == '__main__' :
    args = parse_args()
    topk_dict = {}
    for dataset_name , c in  zip( ['zero','non_zero'] , ['red','yellow'] ):
        def f():
            plt.figure()
            plt.axis( (0.0075 , 0.0175 , 0.005 , 0.015) )
            dataset = ZeroDataset(config.train['val_img_list'][dataset_name],config.train['attribute_file'],config.train['label_dict_file'], config.train['labelname_to_realname_file'] , config.train['attribute_index'] , is_training = False , has_filename = True)

            dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )
            
            net_name = config.net.pop('name')
            net = eval(net_name)( **config.net )
            net.cuda()
            config.net['name'] = net_name
            last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  ) 
            
            psnr_list = []
            output_list = []

            topk_list = [[],[]]
            tt = time()
            net.eval()
            with torch.no_grad():
                for step , batch in tqdm(enumerate( dataloader ) , total = len(dataloader) ):
                    for k in batch:
                        if 'img' in k:
                            batch[k] = batch[k].cuda(async = True)
                            batch[k].requires_grad = False

                    
                    results = net( batch['img'] , None , True)
                    loss_dict = {}
                    k = config.test['k']
                    fc = results['fc']
                    sum_exp = torch.exp(fc).sum(1)
                    topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
                    for i in range(k):
                        loss_dict['top{}'.format(i+1)] =  topk[:,i] 

                    for loss in loss_dict['top1']:
                        topk_list[0].append(float(loss.cpu().detach().numpy()))
                    for loss in loss_dict['top2']:
                        topk_list[1].append(float(loss.cpu().detach().numpy()))

            with open('topk_{}.list'.format(dataset_name),'w') as fp:
                for top1,top2 in zip( *topk_list ):
                    fp.write('{} {}\n'.format(top1,top2))
            #plt.scatter( topk_list[0] , topk_list[1] , label = dataset_name , s = 0.1 )
            #plt.savefig('topk_{}.png'.format(dataset_name))
            return topk_list
        f()

