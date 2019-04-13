import argparse
import datetime
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name',type= str,choices = ['gcn','tede'])
    parser.add_argument('model_info',type=str)
    parser.set_defaults(use_non_zero_net=False)
    parser.add_argument('--use_non_zero_net',dest='use_non_zero_net',action='store_true')
    parser.add_argument('--output_list',default='../submit/ensemble_out.txt')
    parser.add_argument('--top',type=int,default=None)
    parser.add_argument('--resume_epoch',default='best_zero' )
    
    return parser.parse_args()

def main(args):
    fc = []
    with open(args.model_info) as fp:
        first = True
        a = fp.readlines()
        a = [ i for i in map( lambda x : x.strip().split(' ')  , a ) ]
        #print(a)
        if args.top is not None:
            a.sort( key = lambda x : float(x[-1]) , reverse = False )
            a = a[:args.top]
            print( [ i for i in map( lambda x : float(x[-1]) , a ) ] )
        for l in tqdm(a,leave=False): 
            if args.model_name == 'gcn':
                import test_gcn
                random_seed , resume_non_zero_net , resume , nonzero_acc, zero_acc = l[0] , l[1] , float(l[2]) , float(l[3])
                arg = type('', (), {})()
                arg.resume = resume
                arg.input_list = '../data/test.txt'
                arg.resume_non_zero_net = resume_non_zero_net
                arg.resume_coarse_net = None
                arg.zero_acc = zero_acc
                arg.nonzero_acc = nonzero_acc
                arg.x_tag = 'feat'
                arg.output_list = '../data/output.list'
                arg.batch_size = 256
                arg.has_label = False
                if not args.use_non_zero_net:
                    arg.resume_non_zero_net = None

                import train_config as config
                config.train['split_random_seed'] = random_seed
                config.train['graph_similarity'] = 'custom'
                config.parse_config()
                temp_fc = test_gcn.main( arg , config )['fc']

            elif args.model_name == 'tede':
                import test_tede
                arg = type('', (), {})()
                random_seed , resume , _ , nonzero_acc, zero_acc = l[0] , l[1] , l[2] , float(l[3]) , float(l[4])
                arg.resume = resume
                arg.input_list = '../data/test.txt'
                arg.batch_size = 256
                arg.has_label = False
                arg.output_list = '../data/output.list'
                arg.resume_epoch = args.resume_epoch


                import train_config as config
                config.train['split_random_seed'] = random_seed
                config.net['name'] = 'tede_resnet18'
                config.parse_config()
                feature_layer_dim = 384
                config.net['visual_mlp_kwargs']['out_channels'] = feature_layer_dim
                config.net['semantic_mlp_kwargs']['out_channels'] = feature_layer_dim
                temp_fc = test_tede.main( arg , config )['fc']
                os.system( 'rm {}'.format(arg.output_list) )

            if first:
                fc = temp_fc
                first = False
            else:
                for i in range(len(fc)):
                    for k in fc[i]:
                        fc[i][k] += temp_fc[i][k]




    y = []
    for fc_dict in fc:
        y.append( max( fc_dict , key=fc_dict.get) )
    test_list = open('../data/test.txt').read().strip().split('\n')
    assert len(y) == len(test_list)
    output_list = []
    for label,line in zip(y,test_list):
        output_list.append(line.split('\t')[0].strip().split('/')[-1] +'\t' +label)

    with open(args.output_list,'w') as fp:
        fp.write('\n'.join(output_list) +'\n')
        fp.flush()

if __name__ == '__main__':
    args = parse_args()
    main(args)

