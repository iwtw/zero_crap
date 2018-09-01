import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms

class ZeroDataset(data.Dataset):
    def __init__(self , list_file , attribute_file , label_dict_file , labelname_to_realname_file , attribute_index, is_training , has_label = True , has_filename = False):
        self.is_training= is_training
        self.has_filename = has_filename
        img_list = open(list_file,'r').read().strip().split('\n')
        self.has_label = has_label
        self.attribute_index = attribute_index

        #parse attribute_file

        label_dict_list = open(label_dict_file).read().strip().split('\n')
        self.labelname_to_labelno = {}
        self.labelno_to_labelname = {}
        for line in label_dict_list:
            self.labelname_to_labelno[line.split('\t')[0]] = int(line.split('\t')[1])
            self.labelno_to_labelname[int(line.split('\t')[1])] = line.split('\t')[0]

        with open(labelname_to_realname_file) as fp:
            self.labelname_to_realname = {}
            for line in fp.readlines():
                line = line.strip()
                self.labelname_to_realname[line.split('\t')[0]] = line.split('\t')[1]


        self.attr_list = np.loadtxt( attribute_file , dtype = np.float32 )
        self.num_attributes = len(self.attr_list[0])

        #parse list_file
        if self.has_label:
            self.label_list = [ filename.split('\t')[-1]  for filename in img_list ] 
            self.label_list = [self.labelname_to_labelno[label] for label in self.label_list]
            self.num_classes = len(self.labelname_to_labelno) 
            self.img_list = [ filename.split('\t')[0] for filename in img_list  ]
        else:
            self.img_list = img_list
        
        self.to_tensor = torchvision.transforms.ToTensor()
        #self.mean = [ 0.53889946,  0.4293412 ,  0.37825846]
        #self.std = [ 0.25709282,  0.22477216,  0.21124393]
        self.mean = np.array([ 122.66580474,  114.39779785,  101.50643462])/255
        self.std = np.array([ 58.72520428,  57.74421267,  57.50294789])/255
        self.normalize = torchvision.transforms.Normalize(self.mean,self.std)
        self.flip = torchvision.transforms.RandomHorizontalFlip()
        self.random_crop = torchvision.transforms.RandomCrop((64,64))
        self.center_crop = torchvision.transforms.CenterCrop((64,64))
    def __len__( self ):
        return len( self.img_list )
    def __getitem__( self , idx ):
        img = Image.open( self.img_list[idx] )
        img = img.resize( (69,69) , Image.BICUBIC )
        if self.is_training:
            img = self.flip(img)
            img = self.random_crop( img )
            img = self.normalize( self.to_tensor( img ) )
        else:
            img = self.center_crop( img )
            img = self.normalize( self.to_tensor( img ) )
            
            #img = ( self.to_tensor( img ) - 0.5 ) *2.0
        ret_dict = { 'img' : img  }
        if self.has_label:
            ret_dict['label'] = np.array(self.label_list[idx])
            if self.attribute_index is None:
                ret_dict['attribute'] = self.attr_list[self.label_list[idx]]  
            else:
                #print(self.attribute_index)
                ret_dict['attribute'] = self.attr_list[self.label_list[idx]][self.attribute_index].reshape( 1 )  
        if self.has_filename:
            ret_dict['filename'] = self.img_list[idx]
        return ret_dict
