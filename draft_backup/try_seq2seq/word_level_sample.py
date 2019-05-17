#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:13:33 2019

@author: hua
"""

#from pickle import load
#import pandas as pd

import csv
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

#stories = load(open('/Users/hua/Project/data/amazon-fine-food-reviews/review_dataset.pkl', 'rb'))
#
#print('Loaded Stories %d' % len(stories))
#print(type(stories))
#en =[]
#de =[]type()
#for i in range(100):
#    en.append(stories[i]['story'])
#    de.append(stories[i]['highlights'])
#
#lines = pd.DataFrame()
#lines['en'] = en
#lines['de']= de
#
## Appending SOS andEOS to target data : 
#lines.de = lines.de.apply(lambda x : 'SOS_ '+ x + ' _EOS')

## read file
#lines = pd.read_csv('../data/amazon-fine-food-reviews/food_review.csv')
## remane col
#lines = lines.rename(columns={'text': 'en', 'summary': 'de'})

    
#en = [l[i][0] for i in range(1,len(l))]
#de = [l[i][1] for i in range(1,len(l))]
#
## Appending SOS andEOS to target data : 
## lines.de = lines.de.apply(lambda x : 'SOS_ '+ x + ' _EOS')
#for i in range(lines.shape[0]):
#  x = lines.loc[i,'de']
#  lines.loc[i,'de']= 'SOS_ '+ x + ' _EOS'
#print(lines.head())

#'../data/food_review.csv'

with open("../data/amazon-fine-food-reviews/food_review.csv") as f:
    reader = csv.reader(f)
    l = list(reader)


en = list()
de = list()
for i in range(1,1000): #len(l)
    en.append(l[i][0])
    de.append('SOS_ ' + l[i][1] + ' _EOS' )



# Create word dictionaries :
en_words=set()
for line in en:
    for word in line.split():
        if word not in en_words:
            en_words.add(word)
    
de_words=set()
for line in de:
    for word in line.split():
        if word not in de_words:
            de_words.add(word)
            
# get lengths and sizes :
num_en_words = len(en_words)
num_de_words = len(de_words)

max_en_words_per_sample = max([len(sample.split()) for sample in en])+5
max_de_words_per_sample = max([len(sample.split()) for sample in de])+5

num_en_samples = len(en)
num_de_samples = len(de)

# Get lists of words :
input_words = sorted(list(en_words))
target_words = sorted(list(de_words))

en_token_to_int = dict()
en_int_to_token = dict()

de_token_to_int = dict()
de_int_to_token = dict()

#Tokenizing the words ( Convert them to numbers ) :
for i,token in enumerate(input_words):
    en_token_to_int[token] = i
    en_int_to_token[i]     = token

for i,token in enumerate(target_words):
    de_token_to_int[token] = i
    de_int_to_token[i]     = token

# initiate numpy arrays to hold the data that our seq2seq model will use:
encoder_input_data = np.zeros(
    (num_en_samples, max_en_words_per_sample),
    dtype='float32')
decoder_input_data = np.zeros(
    (num_de_samples, max_de_words_per_sample),
    dtype='float32')
decoder_target_data = np.zeros(
    (num_de_samples, max_de_words_per_sample, num_de_words),
    dtype='float32')

# Process samples, to get input, output, target data:
for i, (input_text, target_text) in enumerate(zip(en, de)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = en_token_to_int[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = de_token_to_int[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, de_token_to_int[word]] = 1
            

#Embedding_layer = Embedding(input_dim = num_words, output_dim = vec_len)




# Defining some constants: 
vec_len       = 300   # Length of the vector that we willl get from the embedding layer
latent_dim    = 1024  # Hidden layers dimension 
dropout_rate  = 0.2   # Rate of the dropout layers
batch_size    = 64    # Batch size
epochs        = 30    # Number of epochs

# Define an input sequence and process it.
# Input layer of the encoder :
encoder_input = Input(shape=(None,))

# Hidden layers of the encoder :
encoder_embedding = Embedding(input_dim = num_en_words, output_dim = vec_len)(encoder_input)
encoder_dropout   = (TimeDistributed(Dropout(rate = dropout_rate)))(encoder_embedding)
encoder_LSTM      = CuDNNLSTM(latent_dim, return_sequences=True)(encoder_dropout)

# Output layer of the encoder :
encoder_LSTM2_layer = CuDNNLSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_LSTM2_layer(encoder_LSTM)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
# Input layer of the decoder :
decoder_input = Input(shape=(None,))

# Hidden layers of the decoder :
decoder_embedding_layer = Embedding(input_dim = num_de_words, output_dim = vec_len)
decoder_embedding = decoder_embedding_layer(decoder_input)

decoder_dropout_layer = (TimeDistributed(Dropout(rate = dropout_rate)))
decoder_dropout = decoder_dropout_layer(decoder_embedding)

decoder_LSTM_layer = CuDNNLSTM(latent_dim, return_sequences=True)
decoder_LSTM = decoder_LSTM_layer(decoder_dropout, initial_state = encoder_states)

decoder_LSTM_2_layer = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
decoder_LSTM_2,_,_ = decoder_LSTM_2_layer(decoder_LSTM)

# Output layer of the decoder :
decoder_dense = Dense(num_de_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_LSTM_2)



# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_input, decoder_input], decoder_outputs)

model.summary()

# Define a checkpoint callback :
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]



num_train_samples = 100 #9000
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data[:num_train_samples,:],
               decoder_input_data[:num_train_samples,:]],
               decoder_target_data[:num_train_samples,:,:],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.08,
          callbacks = callbacks_list)
          
          
          