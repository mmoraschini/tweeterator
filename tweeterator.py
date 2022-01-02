import os
import argparse
import pickle

import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, GRU, Flatten, SpatialDropout1D, SimpleRNN
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.test import gpu_device_name

from data_generator import DataGenerator
from loader import Loader


try:
    if gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
except Exception as e:
    print(e)


def train(input, net_type, latent_dim, window, dropout, batch_size, epochs,
          learning_rate, perc_test, n_hidden, regex_to_remove):

    loader = Loader('vicinitas')
    data = loader.load(input, window=window + 1, regex_to_remove=regex_to_remove)
    data = np.array(data, dtype=object)

    word2int = {}
    int2word = {}
    idx = 0
    for seq in data:
        for word in seq:
            if word not in word2int.keys():
                word2int[word] = idx
                int2word[idx] = word
                idx += 1

    n_phrases = len(data)
    test_idx = np.random.choice(np.arange(n_phrases), int(n_phrases * perc_test), replace=False)
    train_idx = np.setdiff1d(np.arange(n_phrases), test_idx)

    train_data = data[train_idx]
    test_data = data[test_idx]

    if net_type == 'GRU':
        input_layer = Input(shape=(window,), name='input')
        embedding = Embedding(input_dim=len(word2int), output_dim=latent_dim, input_length=window, name='embedding')(input_layer)
        if dropout > 0:
            embedding = SpatialDropout1D(dropout, name='dropout')(embedding)
        
        hidden = GRU(latent_dim, return_sequences=True, name='hl1')(embedding)
        for i in range(n_hidden - 2):
            hidden = GRU(latent_dim, return_sequences=True, name=f'hl{i+2}')(hidden)
        hidden = GRU(latent_dim, name='hl{}'.format(n_hidden))(hidden)
        
        output_layer = Dense(len(word2int), activation='softmax', name='output')(hidden)
        
        model = Model(inputs=input_layer, outputs=output_layer)
    elif net_type == 'RNN':
        model = Sequential()

        model.add(Embedding(input_dim=len(word2int), output_dim=latent_dim, input_length=window, name='embedding'))
        if dropout > 0:
            model.add(SpatialDropout1D(dropout, name='dropout'))

        for i in range(1, n_hidden):
            model.add(SimpleRNN(latent_dim, return_sequences=True, name=f'hl{i}'))
        model.add(SimpleRNN(latent_dim, name=f'hl{n_hidden}'))
        model.add(Dense(len(word2int), activation='softmax', name='output'))

    train_data_generator = DataGenerator(train_data, word2int, window, batch_size)
    test_data_generator = DataGenerator(test_data, word2int, window, batch_size)

    optim_adam = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optim_adam, metrics=['categorical_accuracy'])
    model.fit(train_data_generator, steps_per_epoch=train_data_generator.get_n_steps_in_epoch(),
            validation_data=test_data_generator, validation_steps=test_data_generator.get_n_steps_in_epoch(),
            epochs=epochs)

    return model, word2int, int2word


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a RNN to generate tweets.')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='input file')
    parser.add_argument('--net-type', '-n', type=str, default='RNN',
                        help='neural network type: RNN or GRU')
    parser.add_argument('--latent-dim', '-L', type=int, default=50,
                        help='latent space dimension')
    parser.add_argument('--window', '-w', type=int, default=5,
                        help='scan window dimension')
    parser.add_argument('--dropout', '-d', type=float, default=0,
                        help='fraction of the input units to drop. Set to 0 to disable')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='size of the batches')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.0001,
                        help='larning rate for training the model')
    parser.add_argument('--perc-test', '-t', type=float, default=0.2,
                        help='percent of data to reserve for testing')
    parser.add_argument('--hidden', '-H', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--remove', '-r', nargs='+', default=[],
                        help='regular expressions to remove from input texts')
    parser.add_argument('--output-model-path', '-o', type=str, default='model', required=True,
                        help='path where to save the output model')

    args = parser.parse_args()

    model, w2i, i2w = train(args.input, args.net_type, args.latent_dim, args.window, args.dropout, args.batch_size,
                  args.epochs, args.learning_rate, args.perc_test, args.hidden, args.remove)
    
    output_folder = os.path.join('output', args.output_model_path)
    model.save(output_folder)

    dicts_dump_file = os.path.join('output', args.output_model_path + '.pkl')
    with open(dicts_dump_file, 'wb') as f:
        pickle.dump(w2i, f)
        pickle.dump(i2w, f)