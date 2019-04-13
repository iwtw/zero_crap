import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from numpy import prod
from layers import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''


class ResNet( nn.Module ):
    def __init__(self,
            block,
            num_blocks,
            num_features,
            strides,
            num_classes,
            input_shape ,
            first_kernel_size = 7 ,
            use_batchnorm=True,
            activation_fn=partial(nn.ReLU,inplace=True),
            pre_activation=False ,
            use_maxpool = True ,
            use_avgpool = True ,
            feature_layer_dim=None,
            dropout = 0.0 ):
        super(ResNet,self).__init__()
        self.use_batchnorm = use_batchnorm
        self.activation_fn = activation_fn
        self.pre_activation = pre_activation
        self.use_maxpool = use_maxpool
        self.use_avgpool = use_avgpool
        #assert len(num_features) == 5 
        #assert len(num_blocks) == 4 
        self.conv1 = conv( 3 , num_features[0] , first_kernel_size , strides[0] , first_kernel_size//2 , activation_fn , use_batchnorm = use_batchnorm , bias = False  )
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d( 3,2,1 )

        blocks = []
        #blocks.append( self.build_blocks(block,num_features[0],num_features[0], strides[0] ,num_blocks[0]) )
        for i in range( 0,len(num_blocks)):
            blocks.append( self.build_blocks(block,num_features[i],num_features[i+1] , strides[i+1] , num_blocks[i] ) )
        self.blocks = nn.Sequential( *blocks )
        if self.pre_activation:
            self.post_bn = nn.Sequential( nn.BatchNorm2d( num_features[-1] ) , activation_fn() )

        if self.use_maxpool:
            shape = ( input_shape[0]//2//prod(strides) , input_shape[1]//2//prod(strides) )
        else:
            shape = ( input_shape[0]//prod(strides) , input_shape[1]//prod(strides) )

        if use_avgpool:
            self.avgpool = nn.AvgPool2d([*shape],1)
            shape = 1 * 1 
        else:
            shape =  shape[0] * shape[1]

        '''
        all_conv_layers = []
        all_conv_layers_name = []
        for name , module in self._modules.items():
            all_conv_layers.append( getattr(self,name))
            all_conv_layers_name.append( name )
        self.all_conv_layers = nn.DataParallel( nn.Sequential( *all_conv_layers ) )
        list( map( lambda name:delattr(self,name),all_conv_layers_name) )
        '''

        if feature_layer_dim is not None:
            self.fc1 = nn.Sequential( Flatten() , linear( num_features[-1] * shape , feature_layer_dim , activation_fn = None , pre_activation  = False , use_batchnorm = use_batchnorm) )
        self.dropout = nn.Dropout( dropout )

        '''
        all_conv_layers = []
        all_conv_layers_name = []
        for name , module in self._modules.items():
            all_conv_layers.append( getattr(self,name))
            all_conv_layers_name.append( name )
        self.all_conv_layers = nn.DataParallel( nn.Sequential( *all_conv_layers ) )
        list( map( lambda name:delattr(self,name),all_conv_layers_name) )
        '''


        self.fc2 = linear( feature_layer_dim if feature_layer_dim is not None else num_features[-1] * shape , num_classes , use_batchnorm = False )


        #self.fc2 = ArcLinear( feature_layer_dim if feature_layer_dim is not None else num_features[-1] * shape , num_classes )

    def build_blocks(self,block,in_channels,out_channels,stride,length):
        layers = []
        layers.append( block( in_channels , out_channels , 3 ,  stride , self.use_batchnorm , self.activation_fn ,pre_activation=self.pre_activation ) )
        for i in range(1,length):
            layers.append( block(out_channels,out_channels, 3 , 1 , self.use_batchnorm , self.activation_fn , pre_activation = self.pre_activation ) )
        return nn.Sequential( *layers )

    def forward(self,x):

        
        out = self.conv1(x)
        if self.use_maxpool:
            out = self.maxpool(out)
        out = self.blocks(out)
        if self.pre_activation:
            out = self.post_bn( out )

        if self.use_avgpool:
            out = self.avgpool(out)
        '''
        #out = self.all_conv_layers(x)
        
        '''
        if hasattr( self , 'fc1' ) :
            #feats = self.fc1( out.view(out.shape[0],-1))
            feats = self.fc1( out )
        else:
            feats = out

        #out = self.all_conv_layers(x)

        if hasattr( self , 'fc2') :
            feats = feats.view( feats.shape[0] , -1)
            fc2 = self.fc2( self.dropout(fc1))
            return fc2 , feats
        else:
            return feats
        

