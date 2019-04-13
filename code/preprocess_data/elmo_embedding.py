from allennlp.modules.elmo import batch_to_ids
from allennlp.modules.elmo import Elmo
from nltk.corpus import wordnet
from tqdm import tqdm
import re

label_list = open('../data/label_list_wordnet.txt','r').read().strip().split('\n')
label_list = [ i for i in map( lambda x : x.split('\t')[1] , label_list ) ]

cnt = 0
out_list = []
for label in tqdm(label_list) :
    elmo = Elmo( '/home/wtw/.allennlp/elmo_2x4096_512_2048cnn_2xhighway_options.json' , '/home/wtw/.allennlp/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5' , num_output_representations=1 , dropout=0, requires_grad=False, do_layer_norm=False )
    #examples = wordnet.synset('{}.n.01'.format(label)).examples()
    definition = wordnet.synset('{}.n.01'.format(label)).definition()
    definition = re.split('(\W+)',definition)
    definition = [ i for i in map( lambda x : x.strip() , definition) ]
    definition = [ i for i in filter( lambda x : len(x) > 0 , definition ) ] 
    definition = [label , 'is' ] + definition
    tqdm.write(str(definition) )

    ids = batch_to_ids( [definition] )  
    out = elmo(ids)
    #tqdm.write(str(len(out['elmo_representations'])))
    #tqdm.write(str(out['elmo_representations'][0].shape))
    out_list.append( out['elmo_representations'][0][0][0].detach().numpy() )

with open('../data/elmo_class_wordembeddings.txt' , 'w' ) as fp:
    for label , embs in zip( label_list , out_list ):
        fp.write(label+' ')
        for idx , v in enumerate( embs ):
            fp.write("{:.5f}{}".format(float(v),' ' if idx < len(embs) - 1 else '\n' ) )
    fp.flush()
    
