#wrappers for convenience
import torch.nn as nn
from torch.nn.init import xavier_normal_ , kaiming_normal_
import copy
from functools import partial


def get_weight_init_fn( activation_fn  ):
    """get weight_initialization function according to activation_fn
    Notes
    -------------------------------------
    if activation_fn requires arguments, use partial() to wrap activation_fn
    """
    fn = activation_fn
    ret_fn = None
    if hasattr( activation_fn , 'func' ):
        fn = activation_fn.func

    if  fn == nn.LeakyReLU:
        negative_slope = 0 
        if hasattr( activation_fn , 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr( activation_fn , 'args'):
            if len( activation_fn.args) > 0 :
                negative_slope = activation_fn.args[0]
        ret_fn =  partial( kaiming_normal_ ,  a = negative_slope )
    elif fn == nn.ReLU or fn == nn.PReLU :
        ret_fn =  partial( kaiming_normal_ , a = 0 )
    else:
        ret_fn =  xavier_normal_
    #print(ret_fn)
    return ret_fn

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 , activation_fn= None , use_batchnorm = False , pre_activation = False , bias = True , weight_init_fn = None ):
    """pytorch torch.nn.Conv2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        conv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        #assert not bias
        bias = False

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    conv = nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    try:
        weight_init_fn( conv.weight )
    except:
        print( conv.weight )
    layers.append( conv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , activation_fn = None ,   use_batchnorm = False , pre_activation = False , bias= True , weight_init_fn = None ):
    """pytorch torch.nn.ConvTranspose2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        deconv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))

    """
    if not pre_activation and use_batchnorm:
        #assert not bias
        bias = False

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    deconv = nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    weight_init_fn( deconv.weight )
    layers.append( deconv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

def linear( in_channels , out_channels , activation_fn = None , use_batchnorm = False ,pre_activation = False , bias = True ,weight_init_fn = None):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        linear(3,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        #assert not bias
        bias = False

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm1d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    linear = nn.Linear( in_channels , out_channels ,bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    weight_init_fn( linear.weight )

    layers.append( linear )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm1d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )


class Flatten(nn.Module):
    def __init__( self ):
        super(type(self),self).__init__()
    def forward(self, x ):
        return x.view(x.shape[0],-1)

class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed 
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """
    def __init__(self, in_channels , out_channels , kernel_size , stride = 1 , use_batchnorm = False , activation_fn = partial( nn.ReLU ,  inplace=True ) , last_activation_fn = partial( nn.ReLU , inplace=True ) , pre_activation = False , scaling_factor = 1.0):
        super(BasicBlock, self).__init__()
        bias = False if use_batchnorm else True
        if pre_activation and stride == 2:
            self.shortcut = nn.Sequential( nn.BatchNorm2d(in_channels) , activation_fn() )
            self.conv1 = conv( in_channels , out_channels , kernel_size , stride , kernel_size//2 ,  activation_fn = None , pre_activation = False , use_batchnorm = False , bias = bias )
        else:
            self.shortcut = None
            self.conv1 = conv( in_channels , out_channels , kernel_size , stride , kernel_size//2 ,  activation_fn , pre_activation = pre_activation , use_batchnorm = use_batchnorm , bias = bias)
            

        self.conv2 = conv( out_channels , out_channels , kernel_size , 1 , kernel_size//2 , activation_fn , pre_activation = pre_activation ,  use_batchnorm = use_batchnorm , weight_init_fn = get_weight_init_fn(last_activation_fn) , bias = bias )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm if not pre_activation else False , bias = bias  )
        if not pre_activation and last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor

    def forward(self , x ):
        x = (x if self.shortcut is None else self.shortcut( x ))
        if self.downsample is not None:
            residual = self.downsample( x )
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        #print(out.shape,residual.shape)
        out += residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation( out )

        return out

class BottleneckBlock( nn.Module ):
    def __init__( self , in_channels , out_channels , kernel_size , stride = 1 , use_batchnorm = False , activation_fn = partial( nn.ReLU , inplace = True ) , last_activation_fn = partial( nn.ReLU , inplace = True ) , pre_activation = False , scaling_factor = 1.0):
        super(BottleneckBlock , self).__init__()
        mid_channels = out_channels//4
        bias = False if use_batchnorm else True
        if pre_activation and stride == 2 :
            self.shortcut = nn.Sequential( nn.BatchNorm2d(in_channels) , activation_fn() )
            self.conv1 = conv( in_channels , mid_channels , 1 , 1 , 0 , activation_fn = None , pre_activation = False , use_batchnorm = False , bias = bias )
        else:
            self.shortcut = None
            self.conv1 = conv( in_channels , mid_channels , 1 , 1 , 0 , activation_fn , pre_activation = pre_activation , use_batchnorm = use_batchnorm , bias = bias)
            

        self.conv2 = conv( mid_channels , mid_channels , kernel_size , stride , kernel_size//2 , activation_fn , pre_activation  = pre_activation , use_batchnorm = use_batchnorm , bias = bias)
        self.conv3 = conv( mid_channels , out_channels , 1 , 1 , 0 , activation_fn , pre_activation = pre_activation , use_batchnorm = use_batchnorm , bias = bias )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm if not pre_activation else False , bias = bias  )
        if not pre_activation and last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor

    def forward(self , x ):
        x = (x if self.shortcut is None else self.shortcut( x ))
        if self.downsample is not None:
            residual = self.downsample( x )
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        #print(out.shape,residual.shape)
        out += residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation( out )

        return out


