from typing import List, Tuple
import numpy as np

class DataGenerator(object):
    def __init__(self, data: List[List[Tuple[str, str]]], w2i: dict, pos2i: dict, window: int,
                 batch_size: int, shuffle: bool):
        """
        Generate data from a list of tokenised sentences

        Args:
            data (List[List[str]]): list of tokenised sentences, each token is a word represented by a string
            dictionary (dict): dictionary to convert words to their int representation
            window (int): size of the training window
            batch_size (int): batch size
            shuffle (bool): whether to randomly shuffle data before training and after each epoch
        """
        
        self.batch_size = batch_size
        self.window = window
        self._data_size = len(data)
        self._word_data = np.array([[w2i[tok[0]] for tok in sentence] for sentence in data], dtype=object)
        self._pos_data = np.array([[pos2i[tok[1]] for tok in sentence] for sentence in data], dtype=object)
        self._i = 0
        self._curr_step = 0
        self._n_words = len(w2i)
        self._n_pos = len(pos2i)
        self._shuffle = shuffle
        self._n_steps_in_epoch = self.get_n_steps_in_epoch()

        if self._shuffle:
            shuffle_int = np.random.choice(range(self._data_size), self._data_size, replace=False)
            self._word_data = self._word_data[shuffle_int]
            self._pos_data = self._pos_data[shuffle_int]
    
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
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return next batch of data

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The first element contains the int representation of `batch_size` training sentences of length `window`
                The second element contains the one-hot encoding of the next word of the loaded training sentences
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
                shuffle_int = np.random.choice(range(self._data_size), self._data_size, replace=False)
                self._word_data = self._word_data[shuffle_int]
                self._pos_data = self._pos_data[shuffle_int]

        return [w_X, pos_X], [w_Y, pos_Y]
