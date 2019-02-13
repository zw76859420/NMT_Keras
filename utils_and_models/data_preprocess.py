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

def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def data_cal(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        return len(lines)

# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs

# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

if __name__ == '__main__':
    filename = 'G:\\diabetes\\nmt_keras_my\\spa.txt'
    doc = load_doc(filename)
    pairs = to_pairs(doc)
    clean_pairs = clean_pairs(pairs)
    save_clean_data(clean_pairs, 'G:\\diabetes\\nmt_keras_my\\english-spanish.pkl')
    for i in range(900, 1000):
        print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))