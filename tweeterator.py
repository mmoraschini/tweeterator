import numpy as np
import pandas as pd

from keras.models import Model, Sequential
from keras.layers import Dense, Input, GRU, Flatten, SpatialDropout1D, SimpleRNN
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, adam

from data_generator import DataGenerator

import gensim

from loader import Loader

try:
    from tensorflow.test import gpu_device_name
    
    if gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
except Exception as e:
    print(e)

fname = 'data/vicinitas_user_tweets.xlsx'
words_to_remove = ['#Salvini:']
word_feature_dim = 50
latent_dim = 50
num_tokens = 1000
window = 5
dropout = 0
batch_size = 32
epochs = 100
perc_test = 0.2
n_hidden_layers = 2

w2v = True

loader = Loader('vicinitas')
data = loader.load(fname, window=window+1, words_to_remove=words_to_remove)
data = np.array(data)

word2int = {}
int2word = {}
idx = 0
for seq in data:
    for word in seq:
        if word not in word2int.keys():
            word2int[word] = idx
            int2word[idx] = word
            idx += 1
        else:
            continue

#data_int = np.array([[word2int[word] for word in sentence] for sentence in data])
#data = data_int

n_phrases = len(data)
train_idx = np.random.choice(np.arange(n_phrases), int(n_phrases*perc_test), replace=False)
test_idx = np.setdiff1d(np.arange(n_phrases), train_idx)

train_data = data[train_idx]
test_data = data[test_idx]

#if w2v:
#    w2v_model = gensim.models.Word2Vec(data, min_count=1, size=word_feature_dim, window=window)
#    input_layer = Input(shape=(word_feature_dim,), name='input')
#else:
#    num_words = len(np.unique(data.split()))
#    input_layer = Input(shape=(window,), name='input')
#    input_layer = Embedding(input_dim=num_words, output_dim=word_feature_dim, input_length=window, name='embedding')(input_layer)
#    if dropout > 0:
#        input_layer = SpatialDropout1D(dropout, name='dropout')(input_layer)

#input_layer = Input(shape=(window,), name='input')
#embedding = Embedding(input_dim=word_feature_dim, output_dim=latent_dim, input_length=window, name='embedding')(input_layer)
#if dropout > 0:
#    embedding = SpatialDropout1D(dropout, name='dropout')(embedding)
#
#hidden = GRU(word_feature_dim, return_sequences=True, name='hl1')(embedding)
#for i in range(n_hidden_layers-2):
#    hidden = GRU(word_feature_dim, return_sequences=True, name='hl{}'.format(i+2))(hidden)
#hidden = GRU(word_feature_dim, name='hl{}'.format(n_hidden_layers))(hidden)
#
#output_layer = Dense(len(word2int), activation='softmax', name='output')(hidden)
#
#model = Model(inputs=input_layer, outputs=output_layer)

model = Sequential()

model.add(Embedding(input_dim=word_feature_dim, output_dim=latent_dim, input_length=window, name='embedding'))
if dropout > 0:
    model.add(SpatialDropout1D(dropout, name='dropout'))

for i in range(1, n_hidden_layers):
    model.add(SimpleRNN(word_feature_dim, return_sequences=True, name='hl{}'.format(i)))
model.add(SimpleRNN(word_feature_dim, name='hl{}'.format(n_hidden_layers)))
model.add(Dense(len(word2int), activation='softmax', name='output'))

train_data_generator = DataGenerator(train_data, word2int, word_feature_dim, window, batch_size)
test_data_generator = DataGenerator(test_data, word2int, word_feature_dim, window, batch_size)

optim_adam = adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optim_adam, metrics=['categorical_accuracy'])
model.fit_generator(train_data_generator, steps_per_epoch=train_data_generator.get_n_steps_in_epoch(),
                    validation_data=test_data_generator, validation_steps=test_data_generator.get_n_steps_in_epoch(),
                    epochs=epochs)