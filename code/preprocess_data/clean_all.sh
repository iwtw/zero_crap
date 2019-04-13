#!/bin/sh

python ./preprocess_data/gray_to_RGB.py
python ./preprocess_data/gray_to_RGB_test.py
python ./preprocess_data/del_attributes.py

#python ./preprocess_data/label_list_wordnet.py
cp ../data/semifinal_image_phase2/label_list.txt ../data/label_list_wordnet.txt
python ./preprocess_data/elmo_embedding.py
python ./preprocess_data/add_embedding.py
python ./preprocess_data/split.py
