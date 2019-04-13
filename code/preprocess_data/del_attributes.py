import numpy as np

attr_list = open('../data/semifinal_image_phase2/attributes_per_class.txt').read().strip().split('\n')
NUM_ATTRIBTUES = 50
N = NUM_ATTRIBTUES + 1



attr_value_list = [[] for i in range(N)]
for line in attr_list:
    line = line.split('\t')
    assert len(line) == N
    label = line[0]
    for i in range( 1 , len(line)):
        attr_value_list[i].append( float(line[i]) )

del_list = []
for i in range(1,N):
    tempv = attr_value_list[i][0]
    flag = True
    for v in attr_value_list[i]:
        if v != tempv:
            flag = False
            break
    if flag:
        print(i)
        del_list.append(i)


new_attr_list = []
for line in attr_list:
    line = line.split('\t')
    assert len(line) == N
    label = line[0]
    for i in reversed(range( 1 , len(line))):
        if i in del_list:
            del(line[i])
    new_attr_list.append('\t'.join(line))

with open('../data/attributes_per_class_cleaned.txt','w') as fp:
    fp.write('\n'.join(new_attr_list) + '\n' )
    fp.flush()

