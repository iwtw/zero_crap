import skimage.io as io
import numpy as np
from tqdm import tqdm
import os

img_list = open('../data/semifinal_image_phase2/image.txt').read().strip().split('\n')
os.system('mkdir -p ../data/test_rgb')

out_list = []
for line in tqdm(img_list):
    img_name = line.split('\t')[0]
    img = io.imread( '../data//semifinal_image_phase2/test/' + img_name )
    if len(img.shape)==2:
        img = np.stack( [img,img,img] , axis = 2 )
        io.imsave( '../data/test_rgb/' + img_name , img )
    else:
        os.system('cp {} {}'.format('../data/semifinal_image_phase2/test/' + img_name , '../data/test_rgb/' + img_name))
    out_list.append( '../data/test_rgb/' + img_name  )

with open('../data/test.txt','w') as fp:
    fp.write('\n'.join(out_list) +'\n')
    fp.flush()
