import skimage.io as io
import numpy as np
from tqdm import tqdm
import os

img_list = open('./DatasetA_train_20180813/train.txt').read().strip().split('\n')
for img_name in tqdm(img_list):
    img_name = img_name.split('\t')[0]
    img = io.imread( './DatasetA_train_20180813/train/' + img_name )
    if len(img.shape)==2:
        img = np.stack( [img,img,img] , axis = 2 )
        io.imsave( './DatasetA_train_20180813/train_rgb/' + img_name , img )
    else:
        os.system('cp {} {}'.format('./DatasetA_train_20180813/train/' + img_name , './DatasetA_train_20180813/train_rgb/' + img_name))
