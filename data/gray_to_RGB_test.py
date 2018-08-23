import skimage.io as io
import numpy as np
from tqdm import tqdm
import os

img_list = open('./DatasetA_test_20180813/DatasetA_test/image.txt').read().strip().split('\n')
for img_name in tqdm(img_list):
    img_name = img_name.split('\t')[0]
    img = io.imread( './DatasetA_test_20180813/DatasetA_test/test/' + img_name )
    if len(img.shape)==2:
        img = np.stack( [img,img,img] , axis = 2 )
        io.imsave( './DatasetA_test_20180813/DatasetA_test/test_rgb/' + img_name , img )
    else:
        os.system('cp {} {}'.format('./DatasetA_test_20180813/DatasetA_test/test/' + img_name , './DatasetA_test_20180813/DatasetA_test/test_rgb/' + img_name))
