from resnet import ResNet 
import layers 
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import set_requires_grad
from copy import deepcopy
from numpy import prod
from collections import OrderedDict
from nasnet_layers import *
import math

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
        feature_dim = num_features[-1] if kwargs.get('feature_layer_dim') is None else kwargs['feature_layer_dim']
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

class VisualSemanticLayer( nn.Module ):
    def __init__( self , in_channels , num_features , out_channels , activation_fn , use_batchnorm , dropout ):
        super(type(self),self).__init__()
        fcs= []
        num_features = [in_channels] + num_features 
        for idx,n_feat in enumerate(num_features[:-1]):
            in_c = n_feat
            out_c = num_features[idx+1]
            fcs.append( layers.linear( in_c , out_c  , activation_fn = activation_fn , use_batchnorm = use_batchnorm ) )
            fcs.append( nn.Dropout(dropout) )
        fcs.append( layers.linear( num_features[-1] , out_channels , activation_fn = None , use_batchnorm = False ) ) 
        self.dense = nn.Sequential( *fcs )
    def forward( self , x ):
        return self.dense( x )

class QFSLResNet(nn.Module):
    def __init__( self ,  block , num_blocks , num_features , visual_semantic_layers_kwargs ,  **kwargs ):
        super(type(self),self).__init__()
        self.features = ResNet( block , num_blocks , num_features , **kwargs )
        del( self.features.dropout )
        del( self.features.fc2 )
        self.features = nn.DataParallel( self.features )
        self.dropout = nn.Dropout( kwargs['dropout'] )
        self.vs = VisualSemanticLayer( in_channels = kwargs['feature_layer_dim'] ,  **visual_semantic_layers_kwargs )
    def forward( self , x , class_attributes , use_normalization = True):
        feats = self.features( x )
        feats = feats.view( feats.shape[0] , -1 )
        x = self.dropout(feats)
        x = self.vs( x )
        if use_normalization:
            fc = F.linear( F.normalize(x) , F.normalize( class_attributes ) )
        else:
            fc = F.liear( x , class_attributes )
        return { 'fc':fc  }

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
    def __init__( self , A_hat , in_channels , num_features , out_channels , activation_fn , use_batchnorm = False , fm_mult = 1.0 , last_fc = False):
        super(type(self),self).__init__()
        num_features = [ int(x * fm_mult) for x in num_features]
        self.A_hat = A_hat
        self.A_hat.requires_grad = False
        num_features = [in_channels] + num_features 
        layer_list = []
        for prev_num_feature , num_feature in zip( num_features[:-1] , num_features[1:]):
            layer_list.append( GraphConv( self.A_hat , prev_num_feature,num_feature , activation_fn , use_batchnorm = use_batchnorm , bias = True ) )
        if not last_fc:
            layer_list.append( GraphConv( self.A_hat , num_features[-1] , out_channels , activation_fn = None , use_batchnorm = use_batchnorm , bias = True ) )
        else:
            layer_list.append( layers.linear( num_features[-1] , out_channels , activation_fn = None , use_batchnorm = use_batchnorm , bias = True ) )
        self.layers = nn.Sequential( *layer_list )
    def forward( self , x ):
        return self.layers(x)


class ZeroShotGCN(nn.Module):
    def __init__( self , feature_net , gcn   ):
        super(type(self),self).__init__()
        self.feature_net = feature_net
        set_requires_grad( self.feature_net , False )
        self.gcn = gcn
        self.gcn = self.gcn
    def forward( self , x , class_attributes , use_normalization = True ):
        self.feature_net.eval()
        w = self.gcn( class_attributes )
        results = {}
        if x is not None:
            feature_net_result = self.feature_net( x , None , True )
            if use_normalization:
                results['fc'] =  F.linear( F.normalize(feature_net_result['feature']) , F.normalize(w) )
            else:
                results['fc'] =  F.linear( x , w ) 
            results['feature_net_fc'] = feature_net_result['fc']
            results['feature'] = feature_net_result['feature']
        results['weight'] = w
        return results



#from HSE official code
class EmbedGuiding(nn.Module):
    def __init__(self, scores_in_channels , feature_in_channels , fm_mult = 1.0 ,  init=True):
        super(EmbedGuiding, self).__init__()
        self.fc = nn.Linear(scores_in_channels, int(1024*fm_mult))

        self.conv1024 = nn.Conv2d(in_channels= feature_in_channels + int(1024*fm_mult) , out_channels= int(1024*fm_mult), kernel_size=1)
        self.conv2048= nn.Conv2d(in_channels=int(1024*fm_mult), out_channels=feature_in_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

        if init:
            self._init_weights()

    def forward(self, scores, features):
        '''
        scores: (batch_size, scores_dim)
        features: (batch_size, channels, height, width)
        '''
        batch_size = scores.size(0)
        assert features.size(2) == features.size(3), "feature map w & h must be same: {},{}".format(features.size[1], features.size[2])
        size_fm = features.size(2)
        c = features.size(1)
        '''
        prepare scores
        '''
        # fc -->1024
        s = self.fc(scores)
        # repeat to feature map size
        s = s.repeat(1, size_fm*size_fm)
        s = s.view(batch_size, size_fm, size_fm, -1)
        s = s.permute(0,3,2,1) # (n, c, w, h)
        '''
        embed and learn weights
        '''
        # concate with feature map
        cf = torch.cat((s, features), 1)
        # conv to 1024
        cf = self.conv1024(cf)
        cf = self.tanh(cf)
        # conv to 2048
        cf = self.conv2048(cf)
        
        n_cf = cf.size(0)
        c_cf = cf.size(1)
        w_cf = cf.size(2)
        h_cf = cf.size(3)
        cf = cf.view(n_cf*c_cf, w_cf*h_cf)
        cf = self.softmax(cf)
        prior_weights = cf.view(n_cf, c_cf, w_cf, h_cf)

        '''
        guiding
        '''
        # eltwise product with original feature
        embed_feature = torch.mul(features, prior_weights)
        embed_feature = self.relu(embed_feature)
        return embed_feature

    def _init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()
        
        m = self.conv1024
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()

        m = self.conv2048
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()

class HSEResnet(nn.Module):
    def __init__( self , resnet_kwargs , num_classes , fm_mult = 1.0 ):
        super(type(self),self).__init__()
        trunk_kwargs = deepcopy( resnet_kwargs )
        trunk_kwargs['num_features'] = trunk_kwargs['num_features'][:-1]
        trunk_kwargs['num_blocks'] = trunk_kwargs['num_blocks'][:-1]
        trunk_kwargs['use_avgpool'] = False
        trunk_kwargs['feature_layer_dim'] = None
        self.trunk = ResNet(  **trunk_kwargs )
        del( self.trunk.fc2 )
        self.branch1_raw = self.build_raw_branch( resnet_kwargs )
        feature_dim = resnet_kwargs['feature_layer_dim'] if resnet_kwargs['feature_layer_dim'] is not None else resnet_kwargs['num_features'][-1]
        self.fc1 = nn.Linear( feature_dim , num_classes[0] )
        self.branch2_raw = self.build_raw_branch( resnet_kwargs )
        self.branch2_guided_blocks = self.trunk.build_blocks( resnet_kwargs['block'] , resnet_kwargs['num_features'][-2] , resnet_kwargs['num_features'][-1] , resnet_kwargs['strides'][-1] , resnet_kwargs['num_blocks'][-1]  )
        self.branch2_guided_attention = EmbedGuiding(  num_classes[0] ,  resnet_kwargs['num_features'][-1] , fm_mult = fm_mult )
        shape = ( resnet_kwargs['input_shape'][0] // prod( resnet_kwargs['strides'] )  , resnet_kwargs['input_shape'][1] // prod( resnet_kwargs['strides'] ) ) 
        if resnet_kwargs['use_maxpool']:
            shape = ( shape[0] // 2 , shape[1] // 2 )
        self.branch2_fc_guided = nn.Linear( shape[0]*shape[1]*feature_dim , num_classes[1] )
        self.branch2_fc_raw = nn.Linear( feature_dim , num_classes[1] ) 
        self.branch2_fc_cat = nn.Linear( shape[0]*shape[1]*feature_dim + feature_dim , num_classes[1]  )

    def forward(self , x , class_attribtues , use_normalization = True):
        f_I = self.trunk( x)   
        raw1 = self.branch1_raw( f_I )
        raw1 = raw1.view(raw1.shape[0] , - 1)
        if use_normalization:
            s1 = F.linear(  F.normalize( raw1) , F.normalize( self.fc1.weight ) )
        else:
            s1 = self.fc1( raw1 )
        raw2 = self.branch2_raw( f_I )
        guided2 = self.branch2_guided_blocks( f_I )
        guided2 = self.branch2_guided_attention( s1 , guided2 )
        #print(guided2.shape)
        guided2 = guided2.view( guided2.shape[0] , -1  )
        raw2 = raw2.view(  raw2.shape[0] , -1 )
        if use_normalization:
            s2_g = F.linear( F.normalize( guided2 ) ,  F.normalize( self.branch2_fc_guided.weight)  )
            s2_r = F.linear( F.normalize( raw2 ) , F.normalize( self.branch2_fc_raw.weight)  ) 
            s2_cat = F.linear( F.normalize( torch.cat( [guided2 , raw2] , 1 ) ) , F.normalize( self.branch2_fc_cat.weight ) ) 
        else:
            s2_g = self.branch2_fc_guided( guided2.view( guided2 , -1 ) )
            s2_r = self.branch2_fc_raw( raw2.view( raw2 , -1 ) )
            s2_cat = self.branch2_fc_cat( torch.cat( guided2 , raw2 ) )
        s2_avg = (s2_g + s2_r + s2_cat) / 3
        return {'s1':s1 , 's2':s2_avg }

    def build_raw_branch(self , resnet_kwargs ):
        branch_layers = []
        blocks = self.trunk.build_blocks( resnet_kwargs['block'] , resnet_kwargs['num_features'][-2] , resnet_kwargs['num_features'][-1] , resnet_kwargs['strides'][-1] , resnet_kwargs['num_blocks'][-1] )
        branch_layers.append( blocks )
        shape = ( resnet_kwargs['input_shape'][0] // prod( resnet_kwargs['strides'])  , resnet_kwargs['input_shape'][1] // prod( resnet_kwargs['strides'] ) )  
        if resnet_kwargs['use_maxpool']:
            shape = ( shape[0] // 2 , shape[1] // 2 )
        if resnet_kwargs['use_avgpool']:
            avgpool = nn.AvgPool2d( [*shape] , 1 )
            branch_layers.append( avgpool )
            shape = 1 * 1
        if resnet_kwargs['feature_layer_dim'] is not None:
            fc1 = nn.Sequential( layers.Flatten() , layers.linear( resnet_kwargs['num_features'][-1] * shape , resnet_kwargs['feature_layer_dim'] , activation_fn = None , pre_activation  = False , use_batchnorm = resnet_kwargs['use_batchnorm']) )
            branch_layers.append( fc1 )
        return nn.Sequential( *branch_layers )


class TEDEResNet(nn.Module):
    def __init__( self , num_classes , num_attributes ,  feature_net_kwargs,  visual_mlp_kwargs , semantic_mlp_kwargs ):
        super(type(self),self).__init__()
        self.features = ResNet( **feature_net_kwargs )
        del( self.features.dropout )
        del( self.features.fc2 )
        self.features = nn.DataParallel( self.features )
        assert visual_mlp_kwargs['out_channels'] == semantic_mlp_kwargs['out_channels']

        self.visual_mlp = layers.linear( feature_net_kwargs['num_features'][-1] , visual_mlp_kwargs['out_channels'] , activation_fn = visual_mlp_kwargs['activation_fn'] , use_batchnorm = visual_mlp_kwargs['use_batchnorm'] )

        semantic_mlp_layers = []
        prev_num_channels = num_attributes
        for i in range( semantic_mlp_kwargs['num_layers'] ):
            num_channels = round( ( i + 1 ) * ( visual_mlp_kwargs['out_channels'] - num_attributes ) / semantic_mlp_kwargs['num_layers'] ) + num_attributes
            semantic_mlp_layers.append( layers.linear( prev_num_channels , num_channels , activation_fn = semantic_mlp_kwargs['activation_fn'] if i < semantic_mlp_kwargs['num_layers'] - 1 else semantic_mlp_kwargs['last_activation_fn']  , use_batchnorm = semantic_mlp_kwargs['use_batchnorm'] ) )
            prev_num_channels = num_channels

        self.semantic_mlp = nn.Sequential( *semantic_mlp_layers )
        self.classifier = ArcLinear( visual_mlp_kwargs['out_channels'] , num_classes )
        
    def forward( self , x , semantic_description = None):
        visual_feats = self.features( x  )
        visual_feats = visual_feats.view( visual_feats.shape[0] , -1 )
        latent_visual_feats = self.visual_mlp( visual_feats )
        fc = self.classifier( latent_visual_feats , use_normalization = True)
        if semantic_description is not None:
            latent_semantic_feats = self.semantic_mlp( semantic_description )
            return {'latent_visual_feats':latent_visual_feats , 'latent_semantic_feats':latent_semantic_feats , 'fc':fc }

        return {'latent_visual_feats':latent_visual_feats ,  'fc':fc }


    def forward_attribtues( self , attributes ) :
        return {'latent_semantic_feats':self.semantic_mlp( attributes )}

        
class GDEResNet(nn.Module):
    def __init__( self , num_classes , num_attributes ,  feature_net_kwargs,  visual_mlp_kwargs , semantic_gconv_kwargs ):
        super(type(self),self).__init__()
        self.features = ResNet( **feature_net_kwargs )
        del( self.features.dropout )
        del( self.features.fc2 )
        self.features = nn.DataParallel( self.features )
        assert visual_mlp_kwargs['out_channels'] == semantic_gconv_kwargs['out_channels']

        self.visual_mlp = layers.linear( feature_net_kwargs['num_features'][-1] , visual_mlp_kwargs['out_channels'] , activation_fn = visual_mlp_kwargs['activation_fn'] , use_batchnorm = visual_mlp_kwargs['use_batchnorm'] )

        mid_channels = ( num_attributes + visual_mlp_kwargs['out_channels'] ) //2
        self.semantic_gconv = nn.Sequential( 
               GraphConv( semantic_gconv_kwargs['A'] , num_attributes , mid_channels , activation_fn = semantic_gconv_kwargs['activation_fn'] , use_batchnorm = semantic_gconv_kwargs['use_batchnorm']  ) , 
               GraphConv( semantic_gconv_kwargs['A'] , mid_channels , semantic_gconv_kwargs['out_channels'] , activation_fn = semantic_gconv_kwargs['last_activation_fn'] , use_batchnorm = semantic_gconv_kwargs['use_batchnorm'] )
        )
        self.classifier = ArcLinear( visual_mlp_kwargs['out_channels'] , num_classes )
        
    def forward( self , x , class_attributes = None):
        visual_feats = self.features( x  )
        visual_feats = visual_feats.view( visual_feats.shape[0] , -1 )
        latent_visual_feats = self.visual_mlp( visual_feats )
        fc = self.classifier( latent_visual_feats , use_normalization = True)
        if class_attributes is not None:
            latent_semantic_feats = self.semantic_gconv( class_attributes )
            return {'latent_visual_feats':latent_visual_feats , 'latent_semantic_feats':latent_semantic_feats , 'fc':fc }

        return {'latent_visual_feats':latent_visual_feats ,  'fc':fc }


    def forward_attribtues( self , attributes ) :
        return {'latent_semantic_feats':self.semantic_gconv( attributes )}



class NASNet(nn.Module):
    def __init__(self, num_stem_features, num_normal_cells, filters, scaling, skip_reduction, use_aux=True, input_shape=(64,64),
                 num_classes=1000):
        super(NASNet, self).__init__()
        self.num_normal_cells = num_normal_cells
        self.skip_reduction = skip_reduction
        self.use_aux = use_aux
        self.num_classes = num_classes

        '''
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_stem_features, kernel_size=3, stride=2, bias=False)),
            ('bn', nn.BatchNorm2d(num_stem_features, eps=0.001, momentum=0.1, affine=True))
        ]))
        '''
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_stem_features, kernel_size=3, stride=2, padding=1 ,  bias=False)),
            ('bn', nn.BatchNorm2d(num_stem_features, eps=0.001, momentum=0.1, affine=True))
        ]))

        self.cell_stem_0 = CellStem0(in_channels=num_stem_features,
                                     out_channels=int(filters * scaling ** (-2)))
        self.cell_stem_1 = CellStem1(in_channels_x=int(4 * filters * scaling ** (-2)),
                                     in_channels_h=num_stem_features,
                                     out_channels=int(filters * scaling ** (-1)))

        x_channels = int(4 * filters * scaling ** (-1))
        h_channels = int(4 * filters * scaling ** (-2))
        cell_id = 0
        branch_out_channels = filters
        for i in range(3):
            self.add_module('cell_{:d}'.format(cell_id), FirstCell(
                in_channels_left=h_channels, out_channels_left=branch_out_channels // 2, in_channels_right=x_channels,
                out_channels_right=branch_out_channels))
            cell_id += 1
            h_channels = x_channels
            x_channels = 6 * branch_out_channels  # normal: concat 6 branches
            for _ in range(num_normal_cells - 1):
                self.add_module('cell_{:d}'.format(cell_id), NormalCell(
                    in_channels_left=h_channels, out_channels_left=branch_out_channels, in_channels_right=x_channels,
                    out_channels_right=branch_out_channels))
                h_channels = x_channels
                cell_id += 1
            if i == 1 and self.use_aux:
                self.aux_features = nn.Sequential(
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3),
                                 padding=(2, 2), count_include_pad=False),
                    nn.Conv2d(in_channels=x_channels, out_channels=128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.1, affine=True),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=768,
                              kernel_size=((14 + 2) // 3, (14 + 2) // 3), bias=False),
                    nn.BatchNorm2d(num_features=768, eps=1e-3, momentum=0.1, affine=True),
                    nn.ReLU()
                )
                self.aux_linear = nn.Linear(768, num_classes)
            # scaling
            branch_out_channels *= scaling
            if i < 2:
                self.add_module('reduction_cell_{:d}'.format(i), ReductionCell(
                    in_channels_left=h_channels, out_channels_left=branch_out_channels,
                    in_channels_right=x_channels, out_channels_right=branch_out_channels))
                x_channels = 4 * branch_out_channels  # reduce: concat 4 branches

        self.linear = nn.Linear(x_channels, self.num_classes)  # large: 4032; mobile: 1056

        self.num_params = sum([param.numel() for param in self.parameters()])
        if self.use_aux:
            self.num_params -= sum([param.numel() for param in self.aux_features.parameters()])
            self.num_params -= sum([param.numel() for param in self.aux_linear.parameters()])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_( m.weight )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                #nn.init.kaiming_normal_( m.weight )
                m.bias.data.zero_()

    def features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        prev_x, x = x_stem_0, x_stem_1
        cell_id = 0
        for i in range(3):
            for _ in range(self.num_normal_cells):
                new_x = self._modules['cell_{:d}'.format(cell_id)](x, prev_x)
                prev_x, x = x, new_x
                cell_id += 1
            if i == 1 and self.training and self.use_aux:
                x_aux = self.aux_features(x)
            if i < 2:
                new_x = self._modules['reduction_cell_{:d}'.format(i)](x, prev_x)
                prev_x = x if not self.skip_reduction else prev_x
                x = new_x
        if self.training and self.use_aux:
            return [x, x_aux]
        #return [x]
        return x

    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.avg_pool2d(x, kernel_size=x.size(2)).view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x

    '''
    def forward(self, x):
        x = self.features(x)
        output = self.logits(x[0])
        if self.training and self.use_aux:
            x_aux = x[1].view(x[1].size(0), -1)
            aux_output = self.aux_linear(x_aux)
            return [output, aux_output]
        return [output]
    '''
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = F.avg_pool2d(x, kernel_size=x.size(2)).view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)
        #output = self.logits(x[0])
        return x 


def nasnet(num_classes=1000):
    return NASNet(32, 4, 44, 2, skip_reduction=False, use_aux=False, num_classes=num_classes)


class TEDENasNet(nn.Module):
    def __init__( self , num_classes , num_attributes ,  feature_net_kwargs,  visual_mlp_kwargs , semantic_mlp_kwargs ):
        super(type(self),self).__init__()
        self.features = NASNet( **feature_net_kwargs )
        self.features = nn.DataParallel( self.features )
        assert visual_mlp_kwargs['out_channels'] == semantic_mlp_kwargs['out_channels']

        input_shape = feature_net_kwargs['input_shape']
        feature_net_last_channels = 3072 
        self.visual_mlp = layers.linear( feature_net_last_channels , visual_mlp_kwargs['out_channels'] , activation_fn = visual_mlp_kwargs['activation_fn'] , use_batchnorm = visual_mlp_kwargs['use_batchnorm'] )

        mid_channels = ( num_attributes + visual_mlp_kwargs['out_channels'] ) //2
        semantic_mlp_layers = []
        prev_num_channels = num_attributes
        for i in range( semantic_mlp_kwargs['num_layers'] ):
            num_channels = round( ( i + 1 ) * ( visual_mlp_kwargs['out_channels'] - num_attributes ) / semantic_mlp_kwargs['num_layers'] ) + num_attributes
            semantic_mlp_layers.append( layers.linear( prev_num_channels , num_channels , activation_fn = semantic_mlp_kwargs['activation_fn'] if i < semantic_mlp_kwargs['num_layers'] - 1 else semantic_mlp_kwargs['last_activation_fn']  , use_batchnorm = semantic_mlp_kwargs['use_batchnorm'] ) )
            prev_num_channels = num_channels

        self.semantic_mlp = nn.Sequential( *semantic_mlp_layers )


        self.classifier = ArcLinear( visual_mlp_kwargs['out_channels'] , num_classes )
        
    def forward( self , x , semantic_description = None):
        visual_feats = self.features( x  )
        #print(visual_feats.shape)

        visual_feats = visual_feats.view( visual_feats.shape[0] , -1 )
        latent_visual_feats = self.visual_mlp( visual_feats )
        fc = self.classifier( latent_visual_feats , use_normalization = True)
        if semantic_description is not None:
            latent_semantic_feats = self.semantic_mlp( semantic_description )
            return {'latent_visual_feats':latent_visual_feats , 'latent_semantic_feats':latent_semantic_feats , 'fc':fc }

        return {'latent_visual_feats':latent_visual_feats ,  'fc':fc }


    def forward_attribtues( self , attributes ) :
        return {'latent_semantic_feats':self.semantic_mlp( attributes )}

def tede_nasnet(**kwargs):
    kwargs.pop('input_shape')
    return TEDENasNet( **kwargs )

def arc_resnet10(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2]
    return ArcResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

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

def arc_resnet50(fm_mult,**kwargs):
    feature_layer_dim = [64,256,512,1024,2048]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [3,4,6,3]
    return ArcResNet(layers.BottleneckBlock, num_blocks , feature_layer_dim ,  **kwargs)

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

def qfsl_resnet18(fm_mult,visual_semantic_layers_kwargs,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return QFSLResNet(layers.BasicBlock, num_blocks , feature_layer_dim , visual_semantic_layers_kwargs ,   **kwargs)

def hse_resnet18(resnet_kwargs,**kwargs):
    resnet_fm_mult = resnet_kwargs.pop('fm_mult')
    num_features = [64,64,128,256,512]
    num_features = [ int(num_feature * resnet_fm_mult) for num_feature in num_features ]
    num_blocks = [2,2,2,2]
    resnet_kwargs['num_features'] = num_features
    resnet_kwargs['num_blocks'] = num_blocks
    resnet_kwargs['block'] = layers.BasicBlock
    return HSEResnet(resnet_kwargs,**kwargs)

def tede_resnet18(feature_net_kwargs,**kwargs):
    resnet_fm_mult = feature_net_kwargs.pop('fm_mult')
    num_features = [64,64,128,256,512]
    num_features = [ int(num_feature * resnet_fm_mult) for num_feature in num_features ]
    num_blocks = [2,2,2,2]
    feature_net_kwargs['num_features'] = num_features
    feature_net_kwargs['num_blocks'] = num_blocks
    feature_net_kwargs['block'] = layers.BasicBlock
    kwargs.pop('input_shape')
    return TEDEResNet(feature_net_kwargs = feature_net_kwargs  ,**kwargs)

def tede_resnet50(feature_net_kwargs,**kwargs):
    fm_mult = feature_net_kwargs.pop('fm_mult')
    num_features = [64,256,512,1024,2048]
    num_features = [ int(num_feature * fm_mult) for num_feature in num_features ]
    num_blocks = [3,4,6,3]
    feature_net_kwargs['num_features'] = num_features
    feature_net_kwargs['num_blocks'] = num_blocks
    feature_net_kwargs['block'] = layers.BottleneckBlock
    kwargs.pop('input_shape')
    return TEDEResNet(feature_net_kwargs = feature_net_kwargs  ,**kwargs)

def gde_resnet18(feature_net_kwargs,**kwargs):
    resnet_fm_mult = feature_net_kwargs.pop('fm_mult')
    num_features = [64,64,128,256,512]
    num_features = [ int(num_feature * resnet_fm_mult) for num_feature in num_features ]
    num_blocks = [2,2,2,2]
    feature_net_kwargs['num_features'] = num_features
    feature_net_kwargs['num_blocks'] = num_blocks
    feature_net_kwargs['block'] = layers.BasicBlock
    kwargs.pop('input_shape')
    return GDEResNet(feature_net_kwargs = feature_net_kwargs  ,**kwargs)
