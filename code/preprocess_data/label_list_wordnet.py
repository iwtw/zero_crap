import os

label_list = open('../data/semifinal_image_phase2/label_list.txt').read().strip().split('\n')
out_list = []


def f(line,name):
    out_list.append( line.split('\t')[0] + '\t' + name )

for line in label_list:
    name = line.split('\t')[1]
    if name == 'remote-control':
        f(line,'remote_control')
    elif name == 'sportscar':
        f(line,'sports_car')
    elif name == 'earphones':
        f(line,'earphone')
    elif name == 'pcb':
        f(line,'circuit_board')
    else:
        out_list.append( line )

with open('../data/label_list_wordnet.txt','w') as fp:
    fp.write('\n'.join(out_list) + '\n')
    fp.flush()
