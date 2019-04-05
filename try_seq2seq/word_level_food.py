#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:13:33 2019

@author: hua
"""

print("Importing library...")
#import pandas as pd
import pickle
import csv
import numpy as np

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils

# load dataset
with open('../data/food_review.csv') as f:
    reader = csv.reader(f)
    l = list(reader)

print("Dataset loaded.")
#n_samples =10000

# running time calculation
import timeit
start = timeit.default_timer()

# encoder and decoder
# Appending SOS andEOS to target data (decoder)
en = [l[i][0] for i in range(1, len(l))] #[0:n_samples]
de = ['SOS_ '+ l[i][1] + ' _EOS' for i in range(1, len(l))]  #[0:n_samples]

max_encoder_seq_length = max([len(txt.split(' ')) for txt in en])
max_decoder_seq_length = max([len(txt.split(' ')) for txt in de])

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

print('num_en_samples: ', num_en_samples)
print('num_de_samples: ', num_de_samples)
print('num_en_words: ',num_en_words)
print('num_de_words: ',num_de_words)
print('max_en_words_per_sample: ', max_en_words_per_sample)
print('max_de_words_per_sample: ', max_de_words_per_sample)


# Tokenize
en_tokenizer = Tokenizer(num_words=max_encoder_seq_length, char_level=False)
en_tokenizer.fit_on_texts(en)

de_tokenizer = Tokenizer(num_words=max_decoder_seq_length, char_level=False)
de_tokenizer.fit_on_texts(de)


source_token = en_tokenizer.texts_to_sequences(en)
target_token = de_tokenizer.texts_to_sequences(de)

# padding
source_padded = pad_sequences(source_token, maxlen=max_encoder_seq_length, padding = "post")
target_padded = pad_sequences(target_token, maxlen=max_decoder_seq_length, padding = "post")
#print(len(source_padded[0]))
#print(len(target_padded[0]))


num_decoder_tokens = num_de_words
n_samples = num_en_samples

'''
prepare data for the LSTM
'''
print("Preparing data...")

X1, X2, y = list(), list(), list()
for i in range(n_samples):
    # generate source sequence
    target = target_padded[i]
    # create padded input target sequence
    target_in = np.insert(target[:-1],0,0)
    # encode
    tar_encoded = utils.to_categorical(target, num_classes=num_decoder_tokens)
    # store
    X2.append(target_in) #tar2_encoded
    y.append(tar_encoded)

# X1 = array(X1)
# X2 = array(X2)
# y = array(y)
encoder_input_data = source_padded #np.array(X1)
decoder_input_data = np.array(X2)
decoder_target_data = np.array(y)

print("encoder_input_data.shape: ", encoder_input_data.shape)
print("decoder_input_data.shape: ", decoder_input_data.shape)
print("decoder_target_data.shape: ", decoder_target_data.shape)

# running time check
stop = timeit.default_timer()
print('Data preparation Runtime: {} s'.format(round(stop - start,2)))



'''
Embedding
'''
# Defining some constants:
vec_len       = 300 #300   # Length of the vector that we will get from the embedding layer
latent_dim    = 1024#1024  # Hidden layers dimension
dropout_rate  = 0.5   # Rate of the dropout layers
batch_size    = 64 #64    # Batch size
epochs        = 100 #30    # Number of epochs

###
num_en_words = len(en_words)
num_de_words = len(de_words)
###

'''
Encoder
'''
print("encoding...")
# Define an input sequence and process it.
# Input layer of the encoder :
encoder_input = Input(shape=(None,))

# Hidden layers of the encoder :
encoder_embedding = Embedding(input_dim = num_en_words, output_dim = vec_len)(encoder_input)
encoder_dropout = (TimeDistributed(Dropout(rate = dropout_rate)))(encoder_embedding)
encoder_LSTM = CuDNNLSTM(latent_dim, return_sequences=True)(encoder_dropout)

# Output layer of the encoder :
encoder_LSTM2_layer = CuDNNLSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_LSTM2_layer(encoder_LSTM)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


'''
Decoder
'''
print("decoding...")
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

'''
Model
'''
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_input, decoder_input], decoder_outputs)

model.summary()

## Define a checkpoint callback :
#checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
#callbacks_list = [checkpoint]

'''
Train the Model
'''
print("Training the model...")
#num_train_samples = 100 #9000
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data,
               decoder_input_data],
               decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.08,
          callbacks = callbacks_list)

# save model
pickle.dump(model, open('seq2seq_model.pkl', 'wb'))

print("Process completed!")

