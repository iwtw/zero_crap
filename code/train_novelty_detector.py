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
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC,SVR
from copy import deepcopy




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume')
    parser.add_argument('--zero_acc',type=float,default=0.124)
    parser.add_argument('--nonzero_acc',type=float,default=0.5)
    parser.add_argument('--batch_size' , type=int,default=256)
    parser.add_argument('--resume_epoch',default=None)
    parser.add_arugment('--verbose')
    parser.set_defaults(versbose=False)
    return parser.parse_args()


def main(config):
    args = parse_args()
        

    zero_class = open( config.train['zero_class_file'] ).read().strip().split('\n')
    #train_list = ['data/split_lists/ZJL_train_splitargs_40_10_0.list' , 'data/split_lists/novelty_train_splitargs_40_10_0.list' ]
    #val_list = ['data/split_lists/ZJL_all_val_splitargs_40_10_0.list' , 'data/split_lists/novelty_val_splitargs_40_10_0.list' ]
    train_list = [ config.train['novelty_train_img_list'] ]
    val_list = [ config.train['novelty_val_img_list']  ]
    kwargs = deepcopy( config.net )
    kwargs.pop('type')
    net_name = kwargs.pop('name')
    for train_input , val_input in zip( train_list , val_list ):
        train_dataset = ZeroDataset(train_input,config , is_training = False , has_label=True,has_filename=True)
        train_dataloader = DataLoader( train_dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )
        val_dataset = ZeroDataset(val_input,config , is_training = False , has_label=True,has_filename=True)
        val_dataloader = DataLoader( val_dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )
        
        net = eval(net_name)( **kwargs )
        net.cuda()
        config.net['name'] = net_name
        last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  ) 
        
        psnr_list = []
        output_list = []
        feat_list = []
        label_list = []
        topk_list = []

        tt = time()
        net.eval()
        with torch.no_grad():
            for step , batch in tqdm(enumerate( train_dataloader ) , total = len(train_dataloader) ):
                for k in batch:
                    if 'img' in k:
                        batch[k] = batch[k].cuda(async = False)
                        batch[k].requires_grad = False
                results = net( batch['img'] , None , True)
                fc = results['fc']
                sum_exp = torch.exp(fc).sum(1)
                topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = 10 , dim = 1 )
                feat_list.append(results['feature'].cpu().detach().numpy())
                label_list.append( batch['label'].cpu().detach().numpy() )
                topk_list.append( topk.detach().cpu().numpy() )

        feats = np.concatenate( feat_list , axis = 0 )
        labels = np.concatenate( label_list , axis = 0 )
        topks = np.concatenate( topk_list , axis = 0 )
        iof = IsolationForest(n_jobs = 8)
        iof.fit( feats )
        svms = {}
        y = np.array([ -1 if train_dataset.labelno_to_labelname[label]  in zero_class else 1 for label in labels ])
        zero_p = sum( y==-1 ) / len(y)
        nonzero_p = sum( y==1 ) / len(y)
        #for x_tag in ['topk','feat']:
        #for c in [1,3]:
        x_tag = 'feat'
        for c in [55,60,65]:
            for kernel in ['rbf']:
                for class_weight in [args.zero_acc * zero_p/(args.nonzero_acc*nonzero_p)]:
                    if x_tag == 'feat':
                        x = feats
                    elif x_tag == 'topk':
                        x = topks
                    #print("fitting {} {} {}".format( c,kernel,class_weight) )
                    svc = SVC( C = c , kernel = kernel , class_weight = {-1:class_weight,1:1} )
                    t = time()
                    svc.fit( x , y )
                    svms['svc_{}_{}_{}_{}'.format(x_tag,c,kernel,class_weight)] = svc
                    #print( 'time : {:.1f}s'.format( time() - t ) )
                    t = time()



        feat_list = []
        label_list = []
        topk_list = []
        with torch.no_grad():
            for step , batch in tqdm(enumerate( val_dataloader ) , total = len(val_dataloader) ):
                for k in batch:
                    if 'img' in k:
                        batch[k] = batch[k].cuda(async = False)
                        batch[k].requires_grad = False
                results = net( batch['img'] , None , True)
                fc = results['fc']
                sum_exp = torch.exp(fc).sum(1)
                topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = 10 , dim = 1 )
                feat_list.append(results['feature'].cpu().detach().numpy())
                label_list.append( batch['label'].cpu().detach().numpy() )
                topk_list.append( topk.detach().cpu().numpy() )
        feats = np.concatenate( feat_list , axis = 0 )
        labels = np.concatenate( label_list , axis = 0 )
        topks = np.concatenate( topk_list , axis = 0 )

        y = np.array( [-1 if train_dataset.labelno_to_labelname[label] in zero_class else 1 for label in labels] )
        def get_acc( y , y_hat , args ):
            diffusion_matrix = np.zeros((2,2),dtype=np.float32)
            for i,j in zip(y,y_hat):
                i = max(i,0)
                j = max(j,0)
                try:
                    diffusion_matrix[i,j] += 1
                except Exception:
                    print(i,j)
                    raise Exception()
            #print( train_input , val_input)
            final_acc = ( diffusion_matrix[0,0] * args.zero_acc + diffusion_matrix[1,1] * args.nonzero_acc ) / len(y) ) 
            if args.verbose:
                print( "novelty acc : {:.2%}".format(np.mean( y == y_hat ) ) )
                print( "inlier acc (non zero ) {:.2%}".format( diffusion_matrix[1,1] / sum(diffusion_matrix[1,:]) ) )
                print( "outlier acc ( zero ) {:.2%}".format( diffusion_matrix[0,0] / sum(diffusion_matrix[0,:]) ) )
                print( "final acc {:.2%}".format( final_acc ) )
                return final_acc
        
        '''
        print('iof')
        y_hat = iof.predict(feats)
        get_acc( y , y_hat , args )
        '''
        best_acc = 0
        best_svm = None
        for k,svm in svms.items():
            x_tag = k.split('_')[1]
            if args.verbose:
                print(k)
            if x_tag == 'feat':
                x = feats
            elif x_tag == 'topk':
                x = topks
            y_hat = svm.predict( x )
            acc = get_acc( y , y_hat , args )
            if best_acc < acc:
                best_acc = acc
                best_svm = svm
        return best_svm


if __name__ == '__main__':
    import train_config as config
    main(config)


        


