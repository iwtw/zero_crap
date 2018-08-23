from resnet import ResNet 
import layers 
import torch.nn as nn
import torch.nn.functional as F
import torch

def get_predict(fc,config,attr_list):
    k = config.test['k']
    sum_exp = torch.exp(fc).sum(1)
    topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
    #trained_mask = topk[:,0] - config.test['delta'] > topk[:,1]
    trained_mask = torch.ones( topk.shape[0] ) >0
    zero_mask = ~trained_mask

    predicts = torch.zeros( fc.shape[0] ).long().cuda()
    #trained
    if len(predicts[trained_mask]) > 0 :
        predicts[trained_mask] = torch.max( fc[trained_mask] , 1 )[1]
    #zero
    if len(predicts[zero_mask]) > 0 :
        print('shit')
        weight = torch.Tensor( [ config.test['pow_base']**(i) for i in range(k) ] ).cuda()
        #print( attr_list_tensor[top_idx].shape )
        attr_list_tensor = torch.FloatTensor( attr_list ).cuda()
        attr = (weight.view(1,-1,1) * attr_list_tensor[ top_idx[zero_mask] ]).sum(1) / weight.sum() 
        dis = torch.norm( attr.view( attr.shape[0] , 1 , -1 ) - attr_list_tensor.view( 1 , attr_list_tensor.shape[0] , -1 ) , p = 2 , dim = 2 )

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
        self.inverse_attribute = InverseAttributeLayer( num_attributes , feature_dim )
    def forward( self , x , attribute_zero , use_normalization = True):
        feats = self.features( x )
        feats = feats.view( feats.shape[0] , -1 )
        fc = self.classifier( feats , use_normalization )
        attribute =  self.attribute( feats ) 
        #feats_ = self.inverse_attribute( attribute )
        #w_zero = self.inverse_attribute( attribute_zero ) 
        #fc_zero = F.linear( F.normalize( feats ) , F.normalize( w_zero ) )

        #return {'fc_all':cat( [ fc, fc_zero ] , dim = 1 ) , 'fc':fc ,  'feature':feats , 'attribute':attribute , 'feature_':feats_ }
        return {'fc':fc ,  'feature':feats , 'attribute':attribute }
        #return {'fc':x , 'feature':feats}



def arc_resnet18(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return ArcResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

