import os
import numpy as np

np.random.seed(0)

NUM_ZERO_CLASS = 20
NUM_VAL_PER_CLASS = 10
img_list_name = './DatasetA_train_20180813/train.txt'
miss_class = [17,20,27,33,74,112,134,136,148,155]
t = [ i for i in range(1,201) ]

for v in miss_class:
    t.remove(v)
zero_class = np.random.choice( np.array(t) , NUM_ZERO_CLASS , replace = False )
#zero_class = np.array( [i for i in range(191,201)] )
print(zero_class)


img_list = open(img_list_name).read().strip().split('\n')
temp_list = []
val_list = []
zero_val_list = []
new_train_list = []
label_cnt = 0 
label_label_no_list = []
for idx , line in enumerate(img_list):
    label = line.split('\t')[1]
    temp_list.append( '/home/wtw/tianchi/zero_shot_image_recognition/data/DatasetA_train_20180813/train_rgb/' + line )
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
            label_label_no_list.append( label + '\t' + str(label_cnt) )
            label_cnt += 1 
        temp_list.clear()

assert label_cnt == 190 -  NUM_ZERO_CLASS

cnt_label_list = [line.split('\t')[0] for line in label_label_no_list]
attr_list = open('./DatasetA_train_20180813/attributes_per_class.txt').read().strip().split('\n')
for line in attr_list:
    label = line.split('\t')[0]
    if label not in cnt_label_list:
        label_label_no_list.append( label + '\t' + str(label_cnt) )
        label_cnt += 1
assert label_cnt == 230



with open('./zero_train.list','w') as fp:
    fp.write('\n'.join(new_train_list) + '\n')
    fp.flush()
with open('./zero_multi_val.list','w') as fp:
    fp.write('\n'.join(val_list) + '\n')
    fp.flush()
with open('./zero_zero_val.list','w') as fp:
    fp.write('\n'.join(zero_val_list)+'\n')
    fp.flush()
with open('labelname_labelno.list','w') as fp:
    fp.write('\n'.join(label_label_no_list)+'\n')
    fp.flush()
os.system('cat zero_multi_val.list zero_zero_val.list > zero_all_val.list')
os.system('python ./sort_class_attributes.py')

