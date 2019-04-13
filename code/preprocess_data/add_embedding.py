import numpy as np

with open('../data/attributes_per_class_cleaned.txt') as fp:
    label_attribute = {}
    attrs = []
    for line in fp.readlines():
        line = line.strip().split('\t')
        label_attribute[line[0]] =  np.array( line[1:] , np.float32 )
        attrs.append( np.array( line[1:] , np.float32 ) )
    '''
    attrs = np.array( attrs )
    mean = np.mean( attrs , axis = 0 )
    std = np.std( attrs , axis = 0 )
    for k,v in label_attribute.items():
        label_attribute[k] = (v - mean ) / std
    '''


def f( label_fname , embs_fname , out_fname ):
    #with open('../data/semifinal_image_phase2/label_list.txt') as fp:
    with open(label_fname) as fp:
        label_list = {}
        for line in fp.readlines():
            line = line.strip().split('\t')
            label_list[line[0]] = line[1] 

    with open(embs_fname) as fp:
        class_wordembeddings = {}
        embs = []
        for line in fp.readlines():
            line = line.strip().split(' ')
            class_wordembeddings[line[0]] =  np.array(line[1:] , np.float32 )
            embs.append(  np.array(line[1:] , np.float32 ) )
        '''
        embs = np.array(embs)
        embs_mean = np.mean( embs ,axis = 0 ) 
        embs_std = np.std( embs ,axis = 0 ) 
        for k,v in class_wordembeddings.items():
            class_wordembeddings[k] = ( v - embs_mean ) / embs_std
        '''
            
        #assert 'goldfish' in class_wordembeddings


    out_label_attribute = {}
    for k in label_attribute:
        out_label_attribute[k] = np.concatenate(  [label_attribute[k].reshape(-1,1)  , class_wordembeddings[label_list[k]].reshape(-1,1) ] , axis = 0 ).reshape( -1 )
        #assert len( label_attribute[k] ) == 350 - 2 

    with open(out_fname,'w') as fp:
        for k in out_label_attribute:
            fp.write( k+'\t' )
            for idx,v in enumerate(out_label_attribute[k]):
                fp.write( '{:.5f}{}'.format(v,'\t' if idx < len(out_label_attribute[k]) -1 else '\n') )
        fp.flush()
                

        

#default glove embeddings
f('../data/semifinal_image_phase2/label_list.txt','../data/semifinal_image_phase2/class_wordembeddings.txt','../data/attributes_per_class_cleaned_plus_glove_class_wordembeddings.txt')

#elmo embeddings
f('../data/label_list_wordnet.txt','../data/elmo_class_wordembeddings.txt','../data/attributes_per_class_cleaned_plus_elmo_class_wordembeddings.txt')


