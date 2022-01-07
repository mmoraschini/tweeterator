import pandas as pd

import text_cleaning as tc

from nltk.tokenize import sent_tokenize
from nltk import download


download('punkt')


class Loader(object):

    def __init__(self, flatten_hashtags=True, flatten_mentions=True) -> None:
        super().__init__()

        self._flatten_hashtags = flatten_hashtags
        self._flatten_mentions = flatten_mentions
            
    def load(self, fname, file_type, text_column, window, regex_to_remove):
        
        if file_type == 'csv':
            tweet_df = pd.read_csv(fname)
        elif file_type == 'excel':
            tweet_df = pd.read_excel(fname)
        else:
            raise RuntimeError('The only possible options for the input file type are \'csv\' or \'excel\'')
        
        texts = tweet_df[text_column]
        texts = texts.str.lower()
        texts = self._clean(texts, regex_to_remove)
        
        data = [] 
        
        for text in texts:
            for sentence in sent_tokenize(text):

                if sentence[-1] == '.':
                    sentence = sentence[:-1]
                
                tokens = sentence.split(' ')
                if len(tokens) < window + 1:
                    continue
                
                sentence_array = [word for word in tokens if len(word) > 0]
                data.append(sentence_array)
        
        return data
    
    def _clean(self, texts_series, regex_to_remove):
        texts_series = texts_series.apply(tc.remove_urls)
        texts_series = texts_series.apply(tc.remove_newlines)
        if len(regex_to_remove) > 0:
            texts_series = texts_series.apply(tc.remove_words(regex_to_remove))
        texts_series = texts_series.apply(str.strip)
        if self._flatten_hashtags:
            texts_series = texts_series.apply(tc.flatten_hashtags)
        if self._flatten_mentions:
            texts_series = texts_series.apply(tc.flatten_mentions)
        texts_series = texts_series.apply(tc.clean_symbols)

        return texts_series