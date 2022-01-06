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
            texts = self._vicinitas_loading(fname, regex_to_remove)
        elif self.loader_type == 'TrackMyHashtag':
            texts = self._trackmyhashtag_loading(fname, regex_to_remove)
        elif self.loader_type == 'text':
            texts = self._text_loading(fname, regex_to_remove)
        else:
            raise(ValueError("The only loader implemented for now is 'vicinitas'"))
        
        data = [] 
        
        for text in texts:
            for sentence in sent_tokenize(text):

                if sentence[-1] == '.':
                    sentence = sentence[:-1]
                
                #tokens = word_tokenize(sentence)
                tokens = sentence.split(' ')
                if len(tokens) < window + 1:
                    continue
                
                sentence_array = [word.lower() for word in tokens if len(word) > 0]
                data.append(sentence_array)
        
        return data
    
    def _vicinitas_loading(self, fname, regex_to_remove):
        tweet_df = pd.read_excel(fname)
        
        texts = tweet_df['Text']
        
        texts = self._clean(texts, regex_to_remove)
        
        return texts
    
    def _trackmyhashtag_loading(self, fname, regex_to_remove):
        tweet_df = pd.read_csv(fname)
        
        texts = tweet_df['Tweet Content']

        texts = self._clean(texts, regex_to_remove)
        
        return texts
    
    def _text_loading(self, fname, regex_to_remove):
        tweet_df = pd.read_csv(fname)
        
        texts = tweet_df['text']

        texts = self._clean(texts, regex_to_remove)
        
        return texts
    
    def _clean(self, texts_series, regex_to_remove):
        texts_series = texts_series.apply(tc.remove_urls)
        texts_series = texts_series.apply(tc.remove_newlines)
        if len(regex_to_remove) > 0:
            texts_series = texts_series.apply(tc.remove_words(regex_to_remove))
        texts_series = texts_series.apply(str.strip)
        texts_series = texts_series.apply(tc.flatten_mentions)
        texts_series = texts_series.apply(tc.flatten_hashtags)
        texts_series = texts_series.apply(tc.clean_symbols)

        return texts_series