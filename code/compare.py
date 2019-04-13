import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('list1')
    parser.add_argument('list2')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    list1 = open(args.list1).read().strip().split('\n')
    list2 = open(args.list2).read().strip().split('\n')
    assert len(list1) == len(list2)
    correct_cnt = 0
    for line1,line2 in zip( list1 , list2 ):
        if line1.split('\t')[1] == line2.split('\t')[1]:
            correct_cnt += 1
    print("{:.3%}".format( correct_cnt / len(list1)) )




