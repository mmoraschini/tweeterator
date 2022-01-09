from typing import List, Tuple, Union
import os
import argparse
import pickle

import numpy as np
import nltk
import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  GRU, SimpleRNN, LSTM, Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History

import data_generator_pos
import data_generator
from loader import Loader
import net


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

try:
    if gpus:
        print('GPU found')
    else:
        print("No GPU found")
except Exception as e:
    print(e)


def train(data: List[List[Tuple[str, str]]], window: int, batch_size: int, epochs: int, perc_val: float,
          shuffle: bool, learning_rate: float, train_two_nets: bool,
          w_net_type: str, w_latent_dim: int, w_n_units: List[int], w_dropout: float, w_n_hidden: int,
          pos_net_type: str, pos_latent_dim: int, pos_n_units: List[int], pos_dropout: float, pos_n_hidden: int) -> Tuple[Union[Model, List[Model]], Union[History, List[History]], dict, dict]:
    """
    Train a Neural Network to predict the next word in a sentence. The network wants to inputs: the preceding words and the preceding POS tags.

    Args:
        data (List[List[Tuple[str, str]]]): list of tokenised sentences, each token is a list of length 2 containing the word and the POS tag
        window (int): size of the training window
        batch_size (int): batch size
        epochs (int): number of epochs to train the model for
        perc_val (float): percentage of input sentences to use for validation
        shuffle (bool): whether to shuffle the input sentences to create variable batches
        learning_rate (float): learning rate
        train_two_nets (bool): whether to train two networks, one to predict the next word and the other to predict the next POS, or only one net to directly predict the word
        w_net_type (str): type of network of the word prediction part, allowed vales are 'RNN', 'GRU' and 'LSTM'
        w_latent_dim (int): number of embeddings in the word prediction part
        w_n_units (List[int]): number of units for each hidden layer of the word prediction part
        w_dropout (float): dropout amount (0 to disable) in the word prediction part
        w_n_hidden (int): number of hidden layers in the word prediction part
        pos_net_type (str): type of network of the POS prediction part, allowed vales are 'RNN', 'GRU' and 'LSTM'
        pos_latent_dim (int): number of embeddings in the POS prediction part
        pos_n_units (List[int]): number of units for each hidden layer in the POS prediction part
        pos_dropout (float): dropout amount (0 to disable) in the POS prediction part
        pos_n_hidden (int): number of hidden layers in the POS prediction part

    Returns:
        Tuple[Union[Model, List[Model]], Union[History, List[History]], dict, dict]:
            the model (or a list with the two models), the training history (or a list with the two histories),
            the dictionaries to convert words to int, int to words, POS to int and int to POS,
            a dictionary containing the training parameters
    """

    if type(w_n_units) not in [np.ndarray, list]:
        w_n_units = [w_n_units]
    
    if type(pos_n_units) not in [np.ndarray, list]:
        pos_n_units = [pos_n_units]
    
    word2int = {}
    int2word = {}
    pos2int = {}
    int2pos = {}
    for sent in data:
        for tok in sent:
            if tok[0] not in word2int.keys():
                idx = len(word2int)
                word2int[tok[0]] = idx
                int2word[idx] = tok[0]
            
            if tok[1] not in pos2int.keys():
                idx = len(pos2int)
                pos2int[tok[1]] = idx
                int2pos[idx] = tok[1]

    n_sentences = len(data)
    test_idx = np.random.choice(np.arange(n_sentences), int(n_sentences * perc_val), replace=False)
    train_idx = np.setdiff1d(np.arange(n_sentences), test_idx)

    train_data = data[train_idx]
    test_data = data[test_idx]

    if w_net_type == 'GRU':
        w_layer = GRU
    elif w_net_type == 'RNN':
        w_layer = SimpleRNN
    elif w_net_type == 'LSTM':
        w_layer = LSTM
    
    if pos_net_type == 'GRU':
        pos_layer = GRU
    elif pos_net_type == 'RNN':
        pos_layer = SimpleRNN
    elif pos_net_type == 'LSTM':
        pos_layer = LSTM

    loss = "categorical_crossentropy"
    
    if train_two_nets:
        w_train_data = [[tok[0] for tok in sentence] for sentence in train_data]
        pos_train_data = [[tok[1] for tok in sentence] for sentence in train_data]
        w_test_data = [[tok[0] for tok in sentence] for sentence in test_data]
        pos_test_data = [[tok[1] for tok in sentence] for sentence in test_data]

        w_model = net.single_branch(window, w_layer, len(word2int), w_latent_dim, w_n_units, w_dropout, w_n_hidden)
        pos_model = net.single_branch(window, pos_layer, len(pos2int), pos_latent_dim, pos_n_units, pos_dropout, pos_n_hidden)

        w_train_data_generator = data_generator.DataGenerator(w_train_data, word2int, window, batch_size, shuffle)
        w_test_data_generator = data_generator.DataGenerator(w_test_data, word2int, window, batch_size, shuffle)

        pos_train_data_generator = data_generator.DataGenerator(pos_train_data, pos2int, window, batch_size, shuffle)
        pos_test_data_generator = data_generator.DataGenerator(pos_test_data, pos2int, window, batch_size, shuffle)

        w_model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate), metrics=['categorical_accuracy'])
        w_history = w_model.fit(w_train_data_generator, steps_per_epoch=w_train_data_generator.get_n_steps_in_epoch(),
                            validation_data=w_test_data_generator, validation_steps=w_test_data_generator.get_n_steps_in_epoch(),
                            epochs=epochs)
        
        pos_model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate), metrics=['categorical_accuracy'])
        pos_history = w_model.fit(pos_train_data_generator, steps_per_epoch=pos_train_data_generator.get_n_steps_in_epoch(),
                            validation_data=pos_test_data_generator, validation_steps=pos_test_data_generator.get_n_steps_in_epoch(),
                            epochs=epochs)
        
        model = [w_model, pos_model]
        history = [w_history, pos_history]
    else:
        train_data_generator = data_generator_pos.DataGenerator(train_data, word2int, pos2int, window, batch_size, shuffle)
        test_data_generator = data_generator_pos.DataGenerator(test_data, word2int, pos2int, window, batch_size, shuffle)

        model = net.double_branch(window, w_layer, len(word2int), w_latent_dim, w_n_units, w_dropout, w_n_hidden,
                      pos_layer, len(pos2int), pos_latent_dim, pos_n_units, pos_dropout, pos_n_hidden)
        model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate), metrics=['categorical_accuracy'])
        history = model.fit(train_data_generator, steps_per_epoch=train_data_generator.get_n_steps_in_epoch(),
                            validation_data=test_data_generator, validation_steps=test_data_generator.get_n_steps_in_epoch(),
                            epochs=epochs)

    conf = {
        'window': window, 'learning_rate': learning_rate, 'perc_test': perc_val, 'batch_size':batch_size, 'epochs': epochs,
        'w_net_type': w_net_type, 'w_latent_dim': w_latent_dim, 'w_n_units': w_n_units, 'w_dropout': w_dropout, 'w_n_hidden': w_n_hidden,
        'pos_net_type': pos_net_type, 'pos_n_units': pos_n_units, 'pos_n_hidden': pos_n_hidden
    }

    dictionaries = {'word2int': word2int, 'int2word': int2word, 'pos2int': pos2int, 'int2pos': int2pos}

    return model, history, dictionaries, conf


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a RNN to generate tweets. When [word prediction, POS prediction] is indicated' + \
        'it means a 2-elements list, the first applies to the word prediction network (or branch), the second to the POS tag prediction.')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='input file')
    parser.add_argument('--file-type', '-t', type=str, required=True,
                        help='Input file type, the possible options are \'csv\' or \'excel\'')
    parser.add_argument('--column', '-c', type=str, default='vicinitas',
                        help='Name of the column of the input file containing the tweets')
    parser.add_argument('--net-type', '-n', nargs='+', default=['RNN', 'RNN'],
                        help='neural network types [word prediction, POS prediction]: RNN, GRU or LSTM')
    parser.add_argument('--latent-dim', '-L', nargs='+', default=[256, 256],
                        help='latent space dimensions [word prediction, POS prediction]')
    parser.add_argument('--n-units-word', '-u', nargs='+', default=[1024],
                        help='number of recurrent units for the word prediction')
    parser.add_argument('--n-units-pos', '-U', nargs='+', default=[1024],
                        help='number of recurrent units for the POS tag prediction')
    parser.add_argument('--window', '-w', type=int, default=5,
                        help='scan window dimension')
    parser.add_argument('--dropout', '-d', nargs='+', default=[0, 0],
                        help='fraction of the input units to drop. Set to 0 to disable [word prediction, POS prediction]')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='size of the batches')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.0001,
                        help='larning rate for training the model')
    parser.add_argument('--perc-test', '-p', type=float, default=0.2,
                        help='percent of data to reserve for testing')
    parser.add_argument('--hidden', '-H', nargs='+', default=[1, 1],
                        help='number of hidden layers [word prediction, POS prediction]')
    parser.add_argument('--remove', '-r', nargs='+', default=[],
                        help='regular expressions to remove from input texts')
    parser.add_argument('--shuffle', '-s', type=bool, default=False,
                        help='whether to shuffle data at the beginning of training and after each epoch')
    parser.add_argument('--output-model-path', '-o', type=str, default='model',
                        help='path where to save the output model')
    parser.add_argument('--train-two-nets', '-T', type=bool, required=True,
                        help='whether to train two networks, one to predict the next word and the other to predict the next POS, or only one net to directly predict the word')

    args = parser.parse_args()

    loader = Loader(args.loader_type)
    data = loader.load(args.input, window=args.window + 1, regex_to_remove=args.remove)
    data = np.array(data, dtype=object)

    nltk.download('averaged_perceptron_tagger')

    data_plus_pos = []
    for sentence in tqdm(data):
        sentence_plus_pos = nltk.pos_tag(sentence)
        sentence_plus_pos = [list(pos) for pos in sentence_plus_pos]
        data_plus_pos.append(sentence_plus_pos)
    data_plus_pos = np.array(data_plus_pos, dtype=object)

    model, history, dictionaries, conf = train(data_plus_pos, args.window, args.batch_size, args.epochs, args.perc_val,
          args.shuffle, args.learning_rate, args.train_two_nets,
          args.net_type[0], args.latent_dim[0], args.n_units_word, args.dropout[0], args.n_hidden[0],
          args.net_type[1], args.latent_dim[1], args.n_units_pos, args.dropout[1], args.n_hidden[1])
    
    output_folder = os.path.join('output', args.output_model_path)
    if type(model) == list:
        model[0].save(os.path.join(output_folder, 'word_model'))
        model[1].save(os.path.join(output_folder, 'pos_model'))

    dicts_dump_file = os.path.join('output', args.output_model_path + '.pkl')
    with open(dicts_dump_file, 'wb') as f:
        pickle.dump(history, f)
        pickle.dump(dictionaries, f)
        pickle.dump(conf, f)
