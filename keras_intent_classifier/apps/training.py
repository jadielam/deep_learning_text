'''
Created on Sep 30, 2016

@author: dearj019
'''

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

import numpy as np

import sys
import json
import os

EMBEDDING_DIM = 100
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
VALIDATION_SPLIT = 0.2

def index_word_vectors(glove_dir):
    embeddings_index = {}
    with open(os.path.join(glove_dir, 'glove.6b.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def read_training_text(folder_path):
    texts = []
    labels = []
    index_labels = {}
    
    for label_name in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, label_name)
        label_index = len(index_labels)
        index_labels[label_index] = label_name
        with open(path) as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                texts.append(line)
                labels.append(label_index)
    
    return texts, labels, index_labels

def tokenize_text(texts, max_sequence_length):
    tokenizer = Tokenizer(nb_words = MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen = max_sequence_length)
    
    return data, word_index

def categorize_labels(labels):
    return to_categorical(np.asarray(labels))

def prepare_embedding_matrix(word_index, embeddings_index):
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, nb_words

def train_model(nb_labels, nb_words, embedding_matrix, max_sequence_length):
    embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM,
                                weights = [embedding_matrix],
                                input_length = max_sequence_length,
                                trainable = False)
    sequence_input = Input(shape = (max_sequence_length,), dtype = 'int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(nb_labels, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
    
    return model        

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    glove_dir = conf['glove_dir']
    text_data_dir = conf['text_data_dir']
    max_sequence_length = conf['max_sequence_length']
    model_output_path = conf['model_output_path']
    
    embeddings_index = index_word_vectors(glove_dir)
    texts, labels, index_labels = read_training_text(text_data_dir)
    data, word_index = tokenize_text(texts, max_sequence_length)
    labels = categorize_labels(labels)
    embedding_matrix, nb_words = prepare_embedding_matrix(word_index, embeddings_index)
    model = train_model(len(index_labels), nb_words, embedding_matrix, max_sequence_length)
    
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch = 2, batch_size = 128)
    model.save_weights(model_output_path, overwrite=True)

if __name__ == "__main__":
    main()