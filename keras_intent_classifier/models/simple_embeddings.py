'''
Created on Sep 30, 2016

@author: dearj019
'''

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

MAX_SEQUENCE_LENGTH = 1000

def model(embedding_layer, no_labels):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation = 'relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation = 'relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation = 'relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    preds = Dense(no_labels, activation = 'softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = 'rmsprop',
                  metrics = ['acc'])
    return model