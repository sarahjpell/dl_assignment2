#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:38:33 2020

@author: sarahpell
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers import RNN
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import keras.callbacks
from keras import Model
import re
from keras.models import load_model
import os 

files = ["256LSTM20_at_epoch0.hd5", "256LSTM20_at_epoch20.hd5"]
#model = load_model('128gru20_at_epoch0.hd5')
f = open('corpus.txt', 'r')
txt = ''
for line in f:
    txt+=line

txt = re.sub(' +', ' ', txt)
txt = txt.lower()

characters = sorted(list(set(txt)))

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

vocab_size = len(characters)
print('Number of unique characters: ', vocab_size)
print(characters)


X = []
Y = []
corp_len = len(txt)
seq_len = 20

for i in range(0, corp_len - seq_len, 1):
    seq = txt[i:i+seq_len]
    label = txt[i + seq_len]
    X.append([char_to_n[char] for char in seq])
    Y.append(char_to_n[label])
    
print('num of extracted seqs: ', len(X))

x_mod = np.reshape(X, (len(X), seq_len, 1))
x_mod = x_mod / float(len(characters))
Y_mod = to_categorical(Y)

# define how model checkpoints are saved
# filepath = "model_weights-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


#final text generation
start = 10   #random row from the X array

# generating characters
for f in files:

    model =load_model(f)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    string_mapped = list(X[start])
    full_string = [n_to_char[value] for value in string_mapped]

    for i in range(500):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))

        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
#        print('seq',seq)
        full_string.append(n_to_char[pred_index])
#        print('full_string',full_string)
        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]
#        print('mapped',string_mapped)
    print('-------------------------------------------------')
    print(f)
    #print(string_mapped)
    j=0
    string = ""
    while j < len(full_string):
        string+=full_string[j]
        j += 1
    print(string)
    print('-------------------------------------------------')




