from typing import List, Tuple, Union
import numpy as np


DataType = Union[List[List[str]],  List[List[Tuple[str, str]]]]

class DataGenerator(object):
    def __init__(self, data: DataType, window: int, batch_size: int, shuffle: bool):
        self.batch_size = batch_size
        self.window = window
        self._shuffle = shuffle
        self._data_size = len(data)
        self._i = 0
        self._curr_step = 0
        self._n_steps_in_epoch = self.get_n_steps_in_epoch()
    
    def __iter__(self):
        return self

    def __len__(self):
        return self._data_size 
    
    def get_n_steps_in_epoch(self) -> int:
        """
        Return how many steps are there in an epoch

        Returns:
            int: number of steps in an epoch
        """
        
        if self._data_size % self.batch_size == 0:
            n_steps = self._data_size // self.batch_size
        else:
            n_steps = self._data_size // self.batch_size + 1
        
        return n_steps


class SingleDataGenerator(DataGenerator):

    def __init__(self, data: List[List[str]], dictionary: dict, window: int, batch_size: int, shuffle: bool):
        """
        Generate data from a list of tokenised sentences

        Args:
            data (List[List[str]]): list of tokenised sentences, each token is a word represented by a string
            dictionary (dict): dictionary to convert words to their int representation
            window (int): size of the training window
            batch_size (int): batch size
            shuffle (bool): whether to randomly shuffle data before training and after each epoch
        """
        super().__init__(data, window, batch_size, shuffle)
        
        self._n_words = len(dictionary)

        self._data = []
        for sentence in data:
            sentence_int = []
            for word in sentence:
                try:
                    sentence_int.append(dictionary[word])
                except KeyError:
                    sentence_int.append(self._n_words)
            self._data.append(sentence_int)

        if self._shuffle:
            np.random.shuffle(self._data)
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return next batch of data

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The first element contains the int representation of `batch_size` training sentences of length `window`
                The second element contains the one-hot encoding of the next word of the loaded training sentences
        """
        
        X = np.empty((self.batch_size, self.window), dtype=np.int)
        Y = np.zeros((self.batch_size, self._n_words), dtype=np.int)
        
        stop = False
        c = 0
        while stop == False:
            
            if self._i >= self._data_size:
                self._i = 0
            
            sentence = self._data[self._i]
            
            n_examples = len(sentence) - self.window
            
            for j in range(n_examples):
                X[c,:] = sentence[j:j+self.window]
                Y[c,sentence[j+self.window]] = 1
                
                c += 1
                
                if c == self.batch_size:
                    stop = True
                    break
            
            self._i += 1
        
        self._curr_step += 1
        if self._curr_step == self._n_steps_in_epoch:
            self._curr_step = 0
            if self._shuffle:
                np.random.shuffle(self._data)

        return X, Y

class DoubleDataGenerator(DataGenerator):
    def __init__(self, data: List[List[Tuple[str, str]]], w2i: dict, pos2i: dict, window: int,
                 batch_size: int, shuffle: bool):
        """
        Generate data from a list of tokenised sentences

        Args:
            data (List[List[Tuple[str, str]]]): list of tokenised sentences, each token is a list of length 2 containing the word and the POS tag
            w2i (dict): dictionary to convert words to their int representation
            pos2i (dict): dictionary to convert POS tags to their int representation
            window (int): size of the training window
            batch_size (int): batch size
            shuffle (bool): whether to randomly shuffle data before training and after each epoch
        """

        super().__init__(data, window, batch_size, shuffle)

        self._n_words = len(w2i)
        self._n_pos = len(pos2i)

        self._word_data = []
        self._pos_data = []
        for sentence in data:
            sent_word_int = []
            sent_pos_int = []
            for tok in sentence:
                try:
                    sent_word_int.append(w2i[tok[0]])
                except KeyError:
                    sent_word_int.append(self._n_words)
                
                try:
                    sent_pos_int.append(pos2i[tok[1]])
                except KeyError:
                    sent_pos_int.append(self._n_pos)
                
            self._word_data.append(sent_word_int)
            self._pos_data.append(sent_pos_int)

        if self._shuffle:
            zipped_list = list(zip(self._word_data, self._pos_data))
            np.random.shuffle(zipped_list)
            self._word_data, self._pos_data = zip(*zipped_list)
    
    def __next__(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return next batch of data

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]:
                The first element contains two np.ndarrays, each containing the int representation of `batch_size` training sentences of length `window`. The first array contains the word data, the second the POS tags data.
                The second element contains two np.ndarrays, each containing the one-hot encoding of the next element of the loaded training sentences. The first array contains the word data, the second the POS tags data.
        """
        
        w_X = np.empty((self.batch_size, self.window), dtype=np.int)
        w_Y = np.zeros((self.batch_size, self._n_words), dtype=np.int)

        pos_X = np.empty((self.batch_size, self.window), dtype=np.int)
        pos_Y = np.zeros((self.batch_size, self._n_pos), dtype=np.int)
        
        stop = False
        c = 0
        while stop == False:
            
            if self._i >= self._data_size:
                self._i = 0
            
            sentence = self._word_data[self._i]
            pos = self._pos_data[self._i]
            
            n_examples = len(sentence) - self.window
            
            for j in range(n_examples):
                w_X[c,:] = sentence[j:j+self.window]
                pos_X[c,:] = pos[j:j+self.window]
                
                w_Y[c,sentence[j+self.window]] = 1
                pos_Y[c,pos[j+self.window]] = 1
                
                c += 1
                
                if c == self.batch_size:
                    stop = True
                    break
            
            self._i += 1
        
        self._curr_step += 1
        if self._curr_step == self._n_steps_in_epoch:
            self._curr_step = 0
            if self._shuffle:
                zipped_list = list(zip(self._word_data, self._pos_data))
                np.random.shuffle(zipped_list)
                self._word_data, self._pos_data = zip(*zipped_list)

        return [w_X, pos_X], [w_Y, pos_Y]
