import torch
import argparse
from network import *
from custom_utils import *
from dataset import ZeroDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model,set_requires_grad
from time import time
import os
from nltk.corpus import wordnet
from copy import deepcopy
from train import compute_loss1
#from sklearn.ensemble import IsolationForest,
#from sklearn.svm import SVC

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume')
    parser.add_argument('-input_list')
    parser.add_argument('--resume_non_zero_net',default=None)
    parser.add_argument('--resume_coarse_net',default=None)
    parser.add_argument('--zero_acc',type=float,default=0.13092)
    parser.add_argument('--nonzero_acc',type=float,default=0.5)
    parser.add_argument('--x_tag',type=str,default='feat')
    #parser.add_argument('-epsilon',type=float)
    parser.add_argument('--output_list',type=str,default='./output.list')
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--has_label',dest='has_label',action='store_true')
    parser.set_defaults(has_label=False)
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
    train_dataloader = DataLoader( train_dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )

    dataset = ZeroDataset( args.input_list , config , is_training = False , has_filename = True , has_label = args.has_label)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )

    novelty_train_dataset = ZeroDataset( config.train['novelty_train_img_list'] , config , is_training = False , has_label=True,has_filename=True)
    novelty_train_dataloader = DataLoader( novelty_train_dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )
    novelty_val_dataset = ZeroDataset( config.train['novelty_val_img_list'],config , is_training = False , has_label=True,has_filename=True)
    novelty_val_dataloader = DataLoader( novelty_val_dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 4 , pin_memory = False )


    A_hat = get_graph( config , config.train['num_all_classes'] , train_dataset.labelno_to_labelname , train_dataset.labelname_to_realname )
    config.net['name'] = 'gcn'
    config.parse_config()
    gcn_kwargs = deepcopy( config.net['gcn'] )
    gcn = GCN( A_hat , **gcn_kwargs )
    feature_net_kwargs = deepcopy( config.net['feature_net'] )
    feature_net_name = feature_net_kwargs.pop('name')
    load_path = feature_net_kwargs.pop('load_path')
    feature_net = eval(feature_net_name)( **feature_net_kwargs )
    #load_model( feature_net , load_path , epoch = 'best_non_zero' ,  strict = True )
    net = ZeroShotGCN( feature_net , gcn )
    net.cuda()
    #feature_net.cuda()
    #set_requires_grad( feature_net , False )

    global label_weight
    label_weight = feature_net.classifier.linear.weight

    #last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  ) 
    last_epoch = load_model( net , args.resume , epoch = 'best_zero'  ) 
    set_requires_grad( net.feature_net , False )

    config.net['name'] = 'arc_resnet18'
    config.net['type'] = 'all'
    config.train['num_zero_classes'] = 30
    config.parse_config()
    kwargs = deepcopy( config.net )
    kwargs['num_attributes'] = 326
    kwargs.pop( 'name' )
    kwargs.pop( 'type' )
    non_zero_net = arc_resnet18( **kwargs )
    if args.resume_non_zero_net is not None:
        load_model( non_zero_net , args.resume_non_zero_net , epoch = 'best_non_zero' )
    non_zero_net.cuda()


    if args.resume_coarse_net is not None:
        config.net['name'] = 'arc_resnet18'
        config.net['type'] = 'coarse'
        config.train['num_zero_classes'] = 30
        config.parse_config()
        kwargs = deepcopy( config.net )
        coarse_net_name = kwargs.pop( 'name' )
        kwargs.pop( 'type' )
        kwargs['feature_layer_dim'] = 256
        kwargs['num_attributes'] = 326
        coarse_net = eval(coarse_net_name)( **kwargs )
        load_model( coarse_net , args.resume_coarse_net , epoch = 'best_zero' )
        coarse_net.cuda()
        coarse_net.eval()
    

    tt = time()
    

    net.eval()
    non_zero_net.eval()

    '''
    def get_novelty_detector(train_dataloader , val_dataloader , net ):
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
        svms = {}
        y = np.array([ -1 if train_dataset.labelno_to_labelname[label]  in zero_class else 1 for label in labels ])
        zero_p = sum( y==-1 ) / len(y)
        nonzero_p = sum( y==1 ) / len(y)
        #class_weight = args.zero_acc * zero_p/(args.nonzero_acc*nonzero_p)
        #svc = SVC( C = c , kernel = kernel , class_weight = {-1:class_weight,1:1} )
        x_tag = args.x_tag
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



        if not args.has_label:
            class_weight = args.zero_acc * zero_p/(args.nonzero_acc*nonzero_p)
            kernel = 'rbf'
            c = 60
            svc = SVC( C = c , kernel = kernel , class_weight = {-1:class_weight,1:1} )
            svc.fit( x , y  )
            return svc
        else:
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
                final_acc = ( diffusion_matrix[0,0] * args.zero_acc + diffusion_matrix[1,1] * args.nonzero_acc ) / len(y)  
                return final_acc
            
            best_acc = 0
            best_svm = None
            best_k = k
            for k,svm in svms.items():
                x_tag = k.split('_')[1]
                if x_tag == 'feat':
                    x = feats
                elif x_tag == 'topk':
                    x = topks
                y_hat = svm.predict( x )
                acc = get_acc( y , y_hat , args )
                if best_acc < acc:
                    best_acc = acc
                    best_svm = svm
                    best_k = k
            print(best_k)
        return best_svm
    '''

    #if args.resume_non_zero_net is not None:
    #    novelty_detector = get_novelty_detector(novelty_train_dataloader , novelty_val_dataloader , non_zero_net )



    feat_list = []
    topk_list = []
    y_zero_list = []
    y_non_zero_list = []
    fc_zero_list = []
    fc_non_zero_list = []

    label_list = []
    filename_list = []
    superclass_diffusion_matrix = np.zeros((10,10),dtype = np.int32)
    cnt = 0 
    with torch.no_grad():
        for step , batch in tqdm(enumerate( dataloader ) , total = len(dataloader) ):
            for k in batch:
                if 'img' in k:
                    batch[k] = batch[k].cuda(async = True)
                    batch[k].requires_grad = False

            
            results = net( batch['img'] , torch.Tensor(dataset.class_attributes).cuda() , True)
            results_non_zero = non_zero_net( batch['img'] , None , True )
            fc = results['fc']
            sum_exp = torch.exp(fc).sum(1)
            topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = 10 , dim = 1 )
            topk_list.append( topk.detach().cpu().numpy()  )
            feat_list.append(results_non_zero['feature'].cpu().detach().numpy())
            fc_non_zero_list.append( results_non_zero['fc'].cpu().detach().numpy() )
            fc_zero_list.append( results['fc'].cpu().detach() )
            if args.resume_coarse_net is None:
                y_zero = get_predict( results , config , dataset.class_attributes , (config.train['num_all_classes'] - config.train['num_test_classes'] - config.train['num_zero_classes'] ,config.train['num_all_classes']  ))
                y_non_zero = get_predict( results_non_zero , config , dataset.class_attributes , (0,config.train['num_all_classes'] - config.train['num_test_classes'] - config.train['num_zero_classes'] ) )
            else:
                '''
                superclass_no = get_predict( coarse_net( batch['img'] , None , True ) , config , dataset.class_attributes ,  class_range = (0,10) )
                #superclass_no = batch['super_class_label']
                real_superclass_no = batch['super_class_label']
                for i,j in zip( real_superclass_no , superclass_no ):
                    superclass_diffusion_matrix[i,j] += 1

                cnt += ( superclass_no == real_superclass_no.cuda() ).long().sum().detach().cpu().numpy()
                class_idx = [] 
                for i in range(superclass_no.shape[0]):
                    class_idx.append( superclass_zero_idx[ superclass_no[i] ])
                y_zero = get_predict2( results , config , dataset.class_attributes , class_idx , (150,230) )
                class_idx = [] 
                for i in range(superclass_no.shape[0]):
                    class_idx.append( superclass_nonzero_idx[ superclass_no[i] ] )
                y_non_zero = get_predict2( results_non_zero , config , dataset.class_attributes , class_idx , (0,150) )
                '''
                pass

            y_zero_list.append( y_zero.cpu().detach().numpy() )
            y_non_zero_list.append(  y_non_zero.cpu().detach().numpy() )

            if args.has_label:
                label_list.append( batch['label'].cpu().detach().numpy() )
            filenames = batch['filename']
            for filename in filenames:
                filename_list.append( filename )

    print( "super_class acc {:.2%}".format(cnt / len(dataset)) )
    print( superclass_diffusion_matrix )
    y_zero = torch.Tensor( np.concatenate( y_zero_list , axis = 0 ) ).cuda()
    y_non_zero = torch.Tensor( np.concatenate( y_non_zero_list , axis = 0 ) ).cuda()
    fc_zero = torch.Tensor( np.concatenate( fc_zero_list , axis = 0 ) ).cuda()
    fc_non_zero = torch.Tensor( np.concatenate( fc_non_zero_list , axis = 0  ) ).cuda()

    feats = np.concatenate( feat_list , axis = 0 )
    if args.has_label:
        label_list = np.concatenate( label_list , axis = 0 )
    topks = np.concatenate( topk_list , axis = 0 )
    topk_delta = topks[:,0] - topks[:,1]
    #y_hat = iof.predict(feats)

    #if args.resume_non_zero_net is None:
    zero_mask = torch.ones( feats.shape[0] ).cuda() > 0 
    #else:
    #    y_hat = novelty_detector.predict( eval(args.x_tag+'s') )
    #    zero_mask = (torch.Tensor( y_hat ) <0).cuda()
    #for epsilon in np.arange(0.000,0.0001,0.00001):
    #    zero_mask = (torch.Tensor(topk_delta) >= epsilon ).cuda()

    non_zero_mask = ~zero_mask
    y = torch.where( zero_mask , y_zero , y_non_zero )
    print( zero_mask.shape , fc_zero.shape , fc_non_zero.shape )
    if args.resume_non_zero_net is None:
        fc = fc_zero
    else:
        fc = torch.where( zero_mask , fc_zero , fc_non_zero )


    output_list = []
    for predict,filename in zip(y,filename_list):
        output_list.append( '{}\t{}'.format(filename.split('/')[-1] , dataset.labelno_to_labelname[int(predict.cpu().detach().numpy())])  )
    
    '''
    with open(args.output_list,'w') as fp:
        fp.write( '\n'.join( output_list ) + '\n' )
    '''


    if args.has_label:
        diffusion_matrix = np.zeros( (2,2) )
        cnt = 0 
        for real_label , label in zip( label_list , non_zero_mask.cpu().detach().numpy() ):
            y = int( not (train_dataset.labelno_to_labelname[real_label]  in zero_class) )
            y_hat = label
            cnt += y == y_hat
            diffusion_matrix[y,y_hat] += 1
        print( cnt / len(label_list) )
        #diffusion_matrix[0,:] /= float(zero_mask.sum())
        #diffusion_matrix[1,:] /= float(non_zero_mask.sum())
        print( diffusion_matrix )
        os.system('python compare.py {} {}'.format(args.output_list , args.input_list) )
    else:
        print('novelty zero proportion:')
        print(sum(zero_mask.long().detach().cpu().numpy()) / zero_mask.shape[0] )

    fc = fc.cpu().detach().numpy()
    fc_list = []
    for single_fc in fc:
        fc_list.append( {train_dataset.labelno_to_labelname[ labelno ] :v for labelno,v in enumerate(single_fc)} )

    cnt = 0
    cnt_false = 0
    new_fc_list = []
    for fcc , yy in zip( fc_list , y ):
        assert args.resume_non_zero_net is None
        if args.resume_non_zero_net is None:
            fccc = { k:fcc[k] for k in (test_class ) }
        else:
            fccc = fcc
        new_fc_list.append(fccc)

        assert train_dataset.labelno_to_labelname[int(yy.detach().cpu().numpy())] in zero_class + test_class

        if  max( fccc  , key = fccc.get ) != train_dataset.labelno_to_labelname[int(yy.detach().cpu().numpy())] :
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
