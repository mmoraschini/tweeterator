import pandas as pd

import text_cleaning as tc

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import download


download('punkt')


class Loader(object):
    
    def __init__(self, loader_type='vicinitas'):
        
        self.loader_type = loader_type
        
    def load(self, fname, window, regex_to_remove):
        
        if self.loader_type == 'vicinitas':
            texts = self.vicinitas_loading(fname, regex_to_remove)
        else:
            raise(ValueError("The only loader implemented for now is 'vicinitas'"))
        
        data = [] 
        
        for text in texts:
            for sentence in sent_tokenize(text):
                
                tokens = word_tokenize(sentence)
                if len(tokens) < window:
                    continue
                
                sentence_array = [word.lower() for word in tokens]
                data.append(sentence_array)
        
        return data
    
    def vicinitas_loading(self, fname, regex_to_remove):
        tweet_df = pd.read_excel(fname)
        
        texts = tweet_df['Text']
        
        texts = texts.apply(tc.remove_urls)
        texts = texts.apply(tc.remove_newlines)
        if len(regex_to_remove) > 0:
            texts = texts.apply(tc.remove_words(regex_to_remove))
        texts = texts.apply(str.strip)
        texts = texts.apply(tc.clean_symbols)
        texts = texts.apply(tc.flatten_mentions)
        
        return texts