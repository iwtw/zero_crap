import skimage.io as io
import numpy as np
from tqdm import tqdm
import os

img_list = open('../data/semifinal_image_phase2/train.txt').read().strip().split('\n')
os.system('mkdir -p ../data/train_rgb')
for line in tqdm(img_list):
    img_name = line.split('\t')[0]
    img = io.imread( '../data/semifinal_image_phase2/train/' + img_name )
    if len(img.shape)==2:
        img = np.stack( [img,img,img] , axis = 2 )
        io.imsave( '../data/train_rgb/' + img_name , img )
    else:
        os.system('cp {} {}'.format('../data/semifinal_image_phase2/train/' + img_name , '../data/train_rgb/' + img_name))

