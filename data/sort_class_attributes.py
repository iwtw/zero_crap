import numpy as np



label_dict_file = 'labelname_labelno.list'
label_dict_list = open(label_dict_file).read().strip().split('\n')
labelname_to_labelno = {}
labelno_to_labelname = {}

for line in label_dict_list:
    labelname_to_labelno[line.split('\t')[0]] = int(line.split('\t')[1])
    labelno_to_labelname[int(line.split('\t')[1])] = line.split('\t')[0]

class_attributes_raw = open('./DatasetA_train_20180813/attributes_per_class_cleaned_plus_class_wordembeddings.txt').read().strip().split('\n')
class_attributes = np.zeros( (len(class_attributes_raw) , 326 ))
for line in class_attributes_raw:
    labelname = line.split('\t')[0]
    labelno = labelname_to_labelno[labelname]
    attribute = np.array( line.split('\t')[1:] )
    class_attributes[labelno] = attribute

np.savetxt('attributes_per_class_cleaned_plus_class_wordembedding_sort_by_labelno.txt', class_attributes , fmt='%.5f' , delimiter='\t' )
