import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('zero_acc',type=float)
    parser.add_argument('non_zero_acc',type=float)
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    args.zero_acc /= 100
    args.non_zero_acc /= 100
    topk_dict = {}
    for dataset_name , c in  zip( ['zero','non_zero'] , ['red','yellow'] ):
        with open('topk_{}.list'.format(dataset_name)) as fp:
            topk_list = [[],[]]
            for line in fp.readlines():
                line = line.split(' ')
                top1 , top2 = float(line[0]) , float(line[1])
                topk_list[0].append(top1)
                topk_list[1].append(top2)
            topk_dict[dataset_name] = topk_list

    for epsilon in np.arange( 0.0 , 6e-3 , 1e-4):
        tot_correct = 0
        print( 'epsilon {}'.format(epsilon))
        correct = {}
        for dataset_name in ['zero','non_zero']:
            correct[dataset_name] = 0
            for top1 , top2 in zip( *topk_dict[dataset_name] ):
                if dataset_name == 'zero':
                    if top1 - top2 < epsilon:
                        correct[dataset_name] += 1
                else:
                    if top1 - top2 >= epsilon:
                        correct[dataset_name] += 1
            tot_correct += correct[dataset_name]
            print('    {}_acc {:.1%}'.format(dataset_name,correct[dataset_name]/len(topk_dict[dataset_name][0])) )
        print( '    {}_acc {:.1%}'.format( 'total' , tot_correct / (len(topk_dict['zero'][0]) + len(topk_dict['non_zero'][0] ) ) ) )
        zero = 7940
        non_zero = 1462
        tot = zero + non_zero
        real_correct = args.zero_acc * correct['zero'] + args.non_zero_acc * correct['non_zero']

        print( ' real_acc {:.1%}'.format(real_correct / tot ))
