import os
import numpy as np

img_list_name = './DatasetA_train_20180813/train.txt'


img_list = open(img_list_name).read().strip().split('\n')
temp_list = []
prev_label = None
val_list = []
new_train_list = []
for idx , line in enumerate(img_list):
    label = line.split('\t')[1]
    if prev_label is not None and  label!= prev_label :
        choice_idx = np.random.choice( np.arange(len(temp_list)) , 10 )
        for idx , v in enumerate(temp_list) :
            if idx in choice_idx or 191<=int(prev_label[3:])<=200:
                val_list.append( v )
            else:
                new_train_list.append( v )
        temp_list.clear()
    temp_list.append( '/home/wtw/tianchi/zero_shot_image_recognition/data/DatasetA_train_20180813/train_rgb/' + line )
    prev_label = label

choice_idx = np.random.choice( np.arange(len(temp_list)) , 10 )
for idx , v in enumerate(temp_list) :
    if idx in choice_idx or 191<=int(label[3:])<=200:
        val_list.append( v )
    else:
        new_train_list.append( v )



with open('./zero_train.list','w') as fp:
    fp.write('\n'.join(new_train_list) + '\n')
    fp.flush()
with open('./zero_val.list','w') as fp:
    fp.write('\n'.join(val_list) + '\n')
    fp.flush()
