import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_zero_classes',type=int,default=30)
    parser.add_argument('-num_val_per_class',type=int,default=10)
    parser.add_argument('-random_seed',type=int,default=0)
    return parser.parse_args()

if __name__ == '__main__':
    os.system('mkdir -p ../data/split_lists')
    args = parse_args()
    split_args = [ args.num_zero_classes , args.num_val_per_class , args.random_seed ]
    np.random.seed(args.random_seed)

    NUM_ZERO_CLASSES = args.num_zero_classes
    NUM_VAL_PER_CLASS = args.num_val_per_class

    img_list_name = '../data/semifinal_image_phase2/train.txt'
    img_list = open(img_list_name).read().strip().split('\n')

    train_classes = {}
    for line in img_list:
        train_classes[line.split('\t')[1].strip()] = 1
    print(len(train_classes))

    t = [ int(k[3:]) for k in train_classes ]
    print(t)

    zero_class = np.random.choice( np.array(t) , NUM_ZERO_CLASSES , replace = False )
    for v in zero_class:
        t.remove(v)
    #zero_class = np.array( [i for i in range(191,201)] )
    print(zero_class)
    with open('../data/split_lists/zero_classes_splitargs_{}_{}_{}'.format(*split_args),'w') as fp:
        temp = [ "ZJL{}".format(i) for i in zero_class ] 
        fp.write('\n'.join(temp) + '\n' )
        fp.flush()

    with open('../data/split_lists/nonzero_classes_splitargs_{}_{}_{}'.format(*split_args),'w') as fp:
        temp = [ "ZJL{}".format(i) for i in t ] 
        fp.write('\n'.join(temp) + '\n' )
        fp.flush()


    #add nonzero classes
    temp_list = []
    val_list = []
    zero_val_list = []
    new_train_list = []
    label_cnt = 0 
    labelname_to_labelno_list = []
    for idx , line in enumerate(img_list):
        label = line.split('\t')[1]
        temp_list.append( '../data/train_rgb/' + line )
        if idx == len(img_list) -1 or label != img_list[idx+1].split('\t')[1] :
            if int(label[3:]) in zero_class:
                zero_val_list += temp_list
            else:
                choice_idx = np.random.choice( np.arange(len(temp_list)) , NUM_VAL_PER_CLASS )
                for idx , v in enumerate(temp_list) :
                    if idx in choice_idx:
                        val_list.append( v )
                    else:
                        new_train_list.append( v ) 
                labelname_to_labelno_list.append( label + '\t' + str(label_cnt) )
                label_cnt += 1 
            temp_list.clear()

    assert label_cnt == len(train_classes) - NUM_ZERO_CLASSES , '{}'.format(label_cnt)

    #add zero classes
    nonzero_label_list = [line.split('\t')[0] for line in labelname_to_labelno_list]
    attr_list = open('../data/semifinal_image_phase2/attributes_per_class.txt').read().strip().split('\n')
    for line in attr_list:
        label = line.split('\t')[0]
        if label not in nonzero_label_list and label in train_classes:
            labelname_to_labelno_list.append( label + '\t' + str(label_cnt) )
            label_cnt += 1
    assert label_cnt == len(train_classes)
    # add test classes
    test_class = []
    for line in attr_list:
        label = line.split('\t')[0]
        if label not in train_classes:
            labelname_to_labelno_list.append( label + '\t' + str(label_cnt) )
            test_class.append(label)
            label_cnt += 1
    assert label_cnt == 225
    
    with open('../data/split_lists/test_classes_splitargs_{}_{}_{}'.format(*split_args),'w') as fp:
        temp = [ i for i in test_class ] 
        fp.write('\n'.join(temp) + '\n' )
        fp.flush()



    with open('../data/split_lists/ZJL_train_splitargs_{}_{}_{}.list'.format(*split_args),'w') as fp:
        fp.write('\n'.join(new_train_list) + '\n')
        fp.flush()
    with open('../data/split_lists/ZJL_nonzero_val_splitargs_{}_{}_{}.list'.format(*split_args),'w') as fp:
        fp.write('\n'.join(val_list) + '\n')
        fp.flush()
    with open('../data/split_lists/ZJL_zero_val_splitargs_{}_{}_{}.list'.format(*split_args),'w') as fp:
        fp.write('\n'.join(zero_val_list)+'\n')
        fp.flush()
    with open('../data/split_lists/labelname_labelno_splitargs_{}_{}_{}.list'.format(*split_args),'w') as fp:
        fp.write('\n'.join(labelname_to_labelno_list)+'\n')
        fp.flush()
    os.system('cat ../data/split_lists/ZJL_nonzero_val_splitargs_{}_{}_{}.list ../data/split_lists/ZJL_zero_val_splitargs_{}_{}_{}.list > ../data/split_lists/ZJL_all_val_splitargs_{}_{}_{}.list'.format( *split_args , *split_args , *split_args ) )
    #os.system('python ./sort_class_attributes.py')

    #sort
    labelname_to_labelno = {}
    labelno_to_labelname = {}

    for line in labelname_to_labelno_list:
        labelname_to_labelno[line.split('\t')[0]] = int(line.split('\t')[1])
        labelno_to_labelname[int(line.split('\t')[1])] = line.split('\t')[0]

    def save_sorted_attributes_plus_class_wordembedding(embedding_type,total_dim):
        class_attributes_raw = open('../data/attributes_per_class_cleaned_plus_{}_class_wordembeddings.txt'.format( embedding_type )).read().strip().split('\n')
        #class_attributes = np.zeros( (len(class_attributes_raw) , 326 ))
        class_attributes = np.zeros( (len(class_attributes_raw) , total_dim ))
        for line in class_attributes_raw:
            labelname = line.split('\t')[0]
            labelno = labelname_to_labelno[labelname]
            attribute = np.array( line.split('\t')[1:] )
            class_attributes[labelno] = attribute

        np.savetxt('../data/split_lists/attributes_per_class_cleaned_plus_{}_class_wordembedding_sort_by_labelno_splitargs_{}_{}_{}.txt'.format(embedding_type , *split_args), class_attributes , fmt='%.5f' , delimiter='\t' )

    def save_sorted_attributes():
        class_attributes_raw = open('../data/attributes_per_class_cleaned.txt').read().strip().split('\n')
        class_attributes = np.zeros( (len(class_attributes_raw) , 49 ))
        for line in class_attributes_raw:
            labelname = line.split('\t')[0]
            labelno = labelname_to_labelno[labelname]
            attribute = np.array( line.split('\t')[1:] )
            class_attributes[labelno] = attribute

        np.savetxt('../data/split_lists/attributes_per_class_cleaned_sort_by_labelno_splitargs_{}_{}_{}.txt'.format(*split_args), class_attributes , fmt='%.5f' , delimiter='\t' )

    save_sorted_attributes()
    save_sorted_attributes_plus_class_wordembedding('glove',49 + 300)
    save_sorted_attributes_plus_class_wordembedding('elmo',49 + 1024)


    def split_novelty_data(a,p):
        n = len(a)
        if n==0:
            return [] , []
        train_idx = np.random.choice( np.arange(n) , int(n*p) , replace = False )
        train_list = []
        val_list = []
        for idx,v in enumerate(a):
            if idx in train_idx:
                train_list.append( v )
            else:
                val_list.append( v )
        return train_list , val_list
    
    def split_novelty_zero_data(a,p):
        n = len(a)
        if n==0:
            return  [], []
        train_class = np.random.choice( np.array( zero_class ) ,  int( NUM_ZERO_CLASSES * p) ) 
        train_list = []
        val_list = []
        for line in a:
            label = line.split('\t')[1].strip()
            if int(label[3:]) in train_class:
                train_list.append( line )
            else :
                val_list.append( line )
        return train_list , val_list



    nonnovelty_train , nonnovelty_val = split_novelty_data(val_list,0.8)
    novelty_train , novelty_val = split_novelty_zero_data(zero_val_list,0.8)
    tot_novelty_train = nonnovelty_train + novelty_train
    tot_novelty_val = nonnovelty_val + novelty_val

    with open('../data/split_lists/novelty_train_splitargs_{}_{}_{}.list'.format(*split_args),'w') as fp:
        fp.write('\n'.join(tot_novelty_train)+'\n')
        fp.flush()
    with open('../data/split_lists/novelty_val_splitargs_{}_{}_{}.list'.format(*split_args),'w') as fp:
        fp.write('\n'.join(tot_novelty_val)+'\n')
        fp.flush()
