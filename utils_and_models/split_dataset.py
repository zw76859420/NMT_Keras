# -*- coding:utf-8 -*-
# author:zhangwei

import string
import math
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from pickle import load
from numpy.random import rand
from numpy.random import shuffle

def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('english-spanish.pkl')
new_dataset = []
shuffle(raw_dataset)
for i in raw_dataset:
    if len(i[1].split(" ")) <= 4 and len(i[0].split(" ")) <= 4:
        new_dataset.append([i[0],i[1]])
new_dataset = array(new_dataset)
n_sentences = len(new_dataset) ######
dataset = new_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
split = math.floor(len(dataset) - (len(dataset)*0.2))
train, test = dataset[:split], dataset[split:]
# save
save_clean_data(dataset, 'G:\\diabetes\\nmt_keras_my\\english-spanish-both.pkl')
save_clean_data(train, 'G:\\diabetes\\nmt_keras_my\\english-spanish-train.pkl')
save_clean_data(test, 'G:\\diabetes\\nmt_keras_my\\english-spanish-test.pkl')