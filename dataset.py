import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms

class ZeroDataset(data.Dataset):
    def __init__(self , list_file , attribute_file , is_training , has_label = True , has_filename = False):
        self.is_training= is_training
        self.has_filename = has_filename
        img_list = open(list_file,'r').read().strip().split('\n')
        self.has_label = has_label

        #parse attribute_file
        attr_list = open( attribute_file ).read().strip().split('\n')
        attr_list = list(map( lambda x:x.split('\t') , attr_list ))
        attr_list.sort(key=lambda x:int(x[0][3:]))
        self.num_attributes = len(attr_list[0]) - 1 

        self.label_dict = {}
        for idx,v in enumerate(attr_list):
            self.label_dict[v[0]] = idx

        self.label_idx_to_label_list = list( map( lambda x:x[0]  , attr_list  ) ) 

        self.attr_list = np.array( list( map( lambda x:x[1:]  , attr_list  ) ) , dtype = np.float32) 

        #parse list_file
        if self.has_label:
            self.label_list = [ filename.split('\t')[-1]  for filename in img_list ] 
            self.label_list = [self.label_dict[label] for label in self.label_list]
            self.num_classes = len(self.label_dict) 
            self.img_list = [ filename.split('\t')[0] for filename in img_list  ]
        else:
            self.img_list = img_list
        
        self.to_tensor = torchvision.transforms.ToTensor()
        #self.mean = [ 0.53889946,  0.4293412 ,  0.37825846]
        #self.std = [ 0.25709282,  0.22477216,  0.21124393]
        self.mean = np.array([ 122.66580474,  114.39779785,  101.50643462])/255
        self.std = np.array([ 58.72520428,  57.74421267,  57.50294789])/255
        self.normalize = torchvision.transforms.Normalize(self.mean,self.std)
    def __len__( self ):
        return len( self.img_list )
    def __getitem__( self , idx ):
        img = Image.open( self.img_list[idx] )
        flip = torchvision.transforms.RandomHorizontalFlip()
        img = flip(img)
        if self.is_training:
            img = self.normalize( self.to_tensor( img ) )
        else:
            img = self.normalize( self.to_tensor( img ) )
            
            #img = ( self.to_tensor( img ) - 0.5 ) *2.0
        ret_dict = { 'img' : img  }
        if self.has_label:
            ret_dict['label'] = np.array(self.label_list[idx])
            ret_dict['attribute'] = self.attr_list[self.label_list[idx]]
        if self.has_filename:
            ret_dict['filename'] = self.img_list[idx]
        return ret_dict
