from resnet import ResNet 
import layers 
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import set_requires_grad

def preprocess_A(A):
    A_tilde = np.eye( A.shape[0] ) + A
    D_tilde = np.sum( A_tilde , axis = 1 ).flatten()
    D_tilde_inv_sqrt = np.diag( D_tilde ** -0.5 )
    A_hat = D_tilde_inv_sqrt.dot( A_tilde ).dot( D_tilde_inv_sqrt )
    return A_hat

def get_predict(results,config,attr_list,mode=1):
    k = config.test['k']

    if mode == 1 :
        out = results['fc']
        trained_mask = torch.ones( out.shape[0] ) >0
    else:
        out = results['attribute']
        trained_mask = -1*torch.ones( out.shape[0] ) >0
    zero_mask = ~trained_mask

    predicts = torch.zeros( out.shape[0] ).long().cuda()
    #trained
    if len(predicts[trained_mask]) > 0 :
        predicts[trained_mask] = torch.max( out[trained_mask] , 1 )[1]
    #zero
    if len(predicts[zero_mask]) > 0 :
        '''
        fc = results['fc']
        sum_exp = torch.exp(fc).sum(1)
        topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        weight = torch.Tensor( [ config.test['pow_base']**(i) for i in range(k) ] ).cuda()
        #print( attr_list_tensor[top_idx].shape )
        attr_list_tensor = torch.FloatTensor( attr_list ).cuda()
        attr = (weight.view(1,-1,1) * attr_list_tensor[ top_idx[zero_mask] ]).sum(1) / weight.sum() 
        dis = torch.norm( attr.view( attr.shape[0] , 1 , -1 ) - attr_list_tensor.view( 1 , attr_list_tensor.shape[0] , -1 ) , p = 2 , dim = 2 )

        predicts[zero_mask] = torch.min( dis, dim = 1 )[1]
        '''
        attr_list_tensor = torch.FloatTensor( attr_list ).cuda()
        attr = results['attribute'] 
        dis = torch.norm( attr.view( attr.shape[0] , 1 , -1  ) - attr_list_tensor.view( 1 , attr_list_tensor.shape[0] , -1 ) , p = 2 , dim = 2 )
        predicts[zero_mask] = torch.min( dis, dim = 1 )[1]

    return predicts


class ArcLinear(nn.Module):
    def __init__( self , in_features , out_features , dropout = 0.0 ):
        super(type(self),self).__init__()
        self.dropout = nn.Dropout( dropout )
        self.linear = nn.Linear( in_features , out_features , bias = False)
        nn.init.xavier_normal_( self.linear.weight )
    def forward( self , x , use_normalization ):
        x = self.dropout( x )
        if use_normalization:
            return F.linear( F.normalize( x ) , F.normalize( self.linear.weight ) )
        else:
            return F.linear( x , self.linear.weight  )

class AttributeLayer(nn.Module):
    def __init__( self , in_channels , out_channels ):
        super(type(self),self).__init__()
        self.fc = layers.linear( in_channels , out_channels )
    def forward( self , x ):
        x = self.fc( x )
        return x

class InverseAttributeLayer(nn.Module):
    def __init__( self , in_channels , out_channels ):
        super(type(self),self).__init__()
        self.fc = layers.linear( in_channels , out_channels )
    def forward( self , x ):
        x = self.fc( x )
        return x


class ArcResNet(nn.Module):
    def __init__( self ,  block , num_blocks , num_features , num_attributes , **kwargs ):
        super(type(self),self).__init__()
        self.features = ResNet( block , num_blocks , num_features , **kwargs )
        del( self.features.dropout )
        del( self.features.fc2 )
        self.features = nn.DataParallel( self.features )
        feature_dim = num_feature[-1] if kwargs.get('feature_layer_dim') is None else kwargs['feature_layer_dim']
        self.classifier = ArcLinear( feature_dim  , kwargs['num_classes']  , dropout = kwargs['dropout'])
        self.attribute = AttributeLayer( feature_dim  , num_attributes)
        #self.inverse_attribute = InverseAttributeLayer( num_attributes , feature_dim )
    def forward( self , x , shit , use_normalization = True):
        feats = self.features( x )
        feats = feats.view( feats.shape[0] , -1 )
        fc = self.classifier( feats , use_normalization )
        attribute =  self.attribute( feats ) 
        return {'fc':fc ,  'feature':feats , 'attribute':attribute }


class RULEResNet(nn.Module):
    def __init__( self ,  block , num_blocks , num_features , **kwargs ):
        super(type(self),self).__init__()
        self.features = ResNet( block , num_blocks , num_features , **kwargs )
        del( self.features.dropout )
        del( self.features.fc2 )
        self.features = nn.DataParallel( self.features )
        self.dropout = nn.Dropout( kwargs['dropout'] )
    def forward( self , x , class_attributes , use_normalization = True):
        feats = self.features( x )
        feats = feats.view( feats.shape[0] , -1 )
        x = self.dropout(feats)
        if use_normalization:
            fc = F.linear( F.normalize(x) , F.normalize( class_attributes ) )
        else:
            fc = F.liear( x , class_attributes )
        return { 'fc':fc , 'attribute':feats }

class RISResNet(nn.Module):
    def __init__( self ,  block , num_blocks , num_features , **kwargs ):
        super(type(self),self).__init__()
        self.features = ResNet( block , num_blocks , num_features , **kwargs )
        del( self.features.dropout )
        del( self.features.fc2 )
        self.features = nn.DataParallel( self.features )
    def forward( self , x , class_attributes , use_normalization = True):
        feats = self.features( x )
        feats = feats.view( feats.shape[0] , -1 )
        return { 'attribute':feats }

class GraphConv(nn.Module ):
    def __init__( self , A , in_channels , out_channels , activation_fn = None, use_batchnorm =False , bias = True):
        super(type(self),self).__init__()
        self.A  = A
        self.linear = nn.Linear( in_channels , out_channels , bias = bias )
        self.W = self.linear.weight
        torch.nn.init.xavier_normal_( self.W )
        if bias:
            self.bias = self.linear.bias
        if use_batchnorm:
            del( self.bias )
            self.bn = nn.BatchNorm1d( out_channels )
        if activation_fn is not None:
            self.activation = activation_fn()
    def forward( self , x ):
        x = torch.matmul( self.A , x ) 
        x = F.linear( x , self.W )
        if hasattr( self  , 'bias' ):
            x += self.bias
        if hasattr( self , 'bn'):
            x = self.bn( x )
        if hasattr(self, 'activation' ):
            x = self.activation( x )
        return x

class GCN(nn.Module):
    def __init__( self , A_hat , in_channels , num_features , out_channels , activation_fn , use_batchnorm = False , fm_mult = 1.0):
        super(type(self),self).__init__()
        num_features = [ int(x * fm_mult) for x in num_features]
        self.A_hat = A_hat
        self.A_hat.requires_grad = False
        num_features = [in_channels] + num_features 
        layers = []
        for prev_num_feature , num_feature in zip( num_features[:-1] , num_features[1:]):
            layers.append( GraphConv( self.A_hat , prev_num_feature,num_feature , activation_fn , use_batchnorm = use_batchnorm , bias = True ) )
        layers.append( GraphConv( self.A_hat , num_features[-1] , out_channels , activation_fn = None , use_batchnorm = use_batchnorm , bias = True ) )
        self.layers = nn.Sequential( *layers )
    def forward( self , x ):
        return self.layers(x)


class ZeroShotGCN(nn.Module):
    def __init__( self , feature_net , gcn   ):
        super(type(self),self).__init__()
        self.features = feature_net
        set_requires_grad( self.features , False )
        self.features.eval()
        self.gcn = gcn
        self.gcn = self.gcn
    def forward( self , x , class_attributes , use_normalization = True ):
        w = self.gcn( class_attributes )
        x = self.features( x )
        if use_normalization:
            return {'fc': F.linear( F.normalize(x) , F.normalize(w) )}
        else:
            return {'fc': F.linear( x , w ) , 'weight':w}



def arc_resnet18(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return ArcResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def arc_resnet34(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [3,4,6,3]
    return ArcResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def rule_resnet18(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return RULEResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def ris_resnet18(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return RISResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def ris_resnet10(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2]
    return RISResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def ris_resnet34(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [3,4,6,3]
    return RISResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

