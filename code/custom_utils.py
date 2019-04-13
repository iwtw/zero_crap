import numpy as np
import torch
import torch.nn.functional as F

def get_graph( config, num_classes , labelno_to_labelname , labelname_to_realname ):
    from nltk.corpus import wordnet
    graph_A = np.zeros( (num_classes , num_classes) )
    for i in range( num_classes ):
        for j in range( i  , num_classes ):
            '''
            attr_x = class_attributes[i]
            attr_y = class_attributes[j]
            '''
            x = wordnet.synset("{}.n.01".format(labelname_to_realname[ labelno_to_labelname[i]]))
            y = wordnet.synset("{}.n.01".format(labelname_to_realname[ labelno_to_labelname[j]]))
            dis = x.shortest_path_distance( y )
            if config.train['graph_similarity'] == 'custom':
                sim = config.train['max_graph_hop'] - dis  + 1
            elif config.train['graph_similarity'] == 'path':
                sim = x.path_similarity( y )
            elif config.train['graph_similarity'] == 'lch':
                sim = x.lch_similarity( y )
            elif config.train['graph_similarity'] == 'wup':
                sim = x.wup_similarity( y )
            #dis = np.linalg.norm( attr_x[26:] - attr_y[26:] ) / np.sqrt(300*(2**2))
            #max_dis = max( [max_dis , dis] )

            #sim = config.train['max_graph_hop'] - dis 
            #sim = 1 - dis
            #super class
            '''
            sim = 0 
            if  all(attr_x[:6] == attr_y[:6]) and sum(attr_x[:6])>0:
                sim +=  15
            #color
            color_dis =  np.linalg.norm( attr_x[6:14] - attr_y[6:14] )
            color_dis_list.append(color_dis)
            if color_dis < 0.5:
                sim += 1.0
            #animal feature
            animal_sim = sum(attr_x[14:18] == attr_y[14:18])
            sim += animal_sim /2

            device_sim = sum( attr_x[18:24] == attr_y[18:24] )
            sim += device_sim /3
            graph_A[i,j] = sim
            graph_A[j,i] = sim
            '''
            if dis <= config.train['max_graph_hop']:
                graph_A[i,j] = sim
                graph_A[j,i] = sim
            else:
                graph_A[i,j] = 0
                graph_A[j,i] = 0
               # graph_A[i,j] = sim 
               # graph_A[j,i] = sim
            if i==j and config.train['graph_similarity'] != 'custom' :
                graph_A[i,j] *= config.train['graph_diagonal']
    #np.savetxt( 'A.txt' , graph_A , fmt="%.2f" )
    A_hat = preprocess_A( graph_A )
    A_hat = torch.FloatTensor( A_hat ).cuda()
    #np.savetxt( 'A_hat.txt' , A_hat.cpu().detach().numpy() , fmt="%.3f" )
    return A_hat

def get_superclass(attribute):
    if sum( attribute[:5] ) > 0:
        super_class_label = int(np.argmax(attribute[:5]))
    else:
        super_class_label = int(attribute[5]) + 5
    super_class_name = ['animal','transportation','clothes','plant','tableware','others','device','building','food','scene']
    return super_class_label
def preprocess_A(A):
    #A_tilde = np.eye( A.shape[0] ) + A
    A_tilde = A
    D_tilde = np.sum( A_tilde , axis = 1 ).flatten()
    D_tilde_inv_sqrt = np.diag( D_tilde ** -0.5 )
    A_hat = D_tilde_inv_sqrt.dot( A_tilde ).dot( D_tilde_inv_sqrt )
    return A_hat

def get_predict(results,config,attr_list,class_range = (0,230) , mode=1):
    k = config.test['k']

    if mode == 1 :
        out = results['fc']
        out = out[:,class_range[0]:class_range[1]]
        predicts = torch.zeros( out.shape[0] ).long().cuda()
        trained_mask = torch.ones( out.shape[0] ) >0
        zero_mask = ~trained_mask
        predicts[trained_mask] = torch.max( out[trained_mask] , 1 )[1] + class_range[0]
    elif mode == 0 :
        out = results['attribute']
        predicts = torch.zeros( out.shape[0] ).long().cuda()
        trained_mask = -1*torch.ones( out.shape[0] ) >0
        zero_mask = ~trained_mask
        attr_list_tensor = torch.FloatTensor( attr_list ).cuda()
        attr = out
        dis = torch.norm( attr.view( attr.shape[0] , 1 , -1  ) - attr_list_tensor.view( 1 , attr_list_tensor.shape[0] , -1 ) , p = 2 , dim = 2 )
        predicts[zero_mask] = torch.min( dis, dim = 1 )[1]
    elif mode == 2:
        latent_feats_list = attr_list
        out = results['latent_visual_feats']
        attr_list_tensor = torch.FloatTensor( attr_list ).cuda()
        attr_list_tensor = attr_list_tensor[ class_range[0]:class_range[1] ]
        attr = out
        predicts = torch.max(  torch.matmul( F.normalize( attr )  , F.normalize( attr_list_tensor ).transpose(0,1) )  , dim = 1 )[1] + class_range[0]

    else:
        raise ValueError('mode value error')


    return predicts

def get_predict2(results, config, attr_list , class_idx , class_range  ):
    out = results['fc']
    predicts = torch.zeros(out.shape[0])
    #print( len(class_idx) , predicts.shape[0] )
    for i in range( len(class_idx) ):
        #print(out[i,class_idx[i]].shape)
        temp = []
        for v in class_idx[i]:
            if class_range[0] <= v < class_range[1] :
                temp.append( v )
        predicts[i] = class_idx[i][ torch.max( out[i][torch.Tensor(temp).long()].view(1,-1) , 1  )[1] ]
    return predicts

