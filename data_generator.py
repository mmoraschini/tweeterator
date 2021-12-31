import numpy as np

class DataGenerator(object):
    def __init__(self, data, dictionary, word_dim, window, batch_size):
        
        self.batch_size = batch_size
        self.window = window
        self.word_dim = word_dim
        self._data_size = len(data)
        self._dictionary = dictionary
        self._data = data
        self._i = 0
        self._n_words = len(dictionary)
    
    def get_n_steps_in_epoch(self):
        
        if self._data_size % self.batch_size == 0:
            n_steps = self._data_size // self.batch_size
        else:
            n_steps = self._data_size // self.batch_size + 1
        
        return n_steps
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        X = np.empty((self.batch_size, self.window), dtype=np.int)
        Y = np.zeros((self.batch_size, self._n_words), dtype=np.int)
        
        stop = False
        c = 0
        while stop == False:
            
            if self._i >= self._data_size:
                self._i = 0
            
            sentence = self._data[self._i]
            
            n_examples = len(sentence) - self.window + 1
            
            for j in range(n_examples):
                X[c,:] = [self._dictionary[word] for word in sentence[j:j+self.window]]
                Y[c,self._dictionary[sentence[j+1]]] = 1
                
                c += 1
                
                if c == self.batch_size:
                    stop = True
                    break
            
            self._i += 1
        
        return X, Y