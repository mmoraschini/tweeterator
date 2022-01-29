from typing import List, Dict

import pandas as pd

import text_cleaning as tc

from nltk.tokenize import sent_tokenize
from nltk import download


download('punkt')


class Loader(object):

    def __init__(self, flatten_hashtags: bool=True, flatten_mentions: bool=True) -> None:
        """
        This class is used to load and preprocess data from an input csv or excel file

        Args:
            flatten_hashtags (bool, optional): If to flatten the hashtags of the input sentences. Defaults to True.
            flatten_mentions (bool, optional): If to flatten the mentions of the input sentences. Defaults to True.
        """
        super().__init__()

        self._flatten_hashtags = flatten_hashtags
        self._flatten_mentions = flatten_mentions
            
    def load(self, fname: str, file_type: str, text_column: str, window: int, regex_replace: Dict[str, str],
             allowed_symbols: List[str]) -> List[List[str]]:
        """
        Load the texts from the specified input file. It is read as a pandas DataFrame and then parsed.

        Args:
            fname (str): name of the file to load
            file_type (str): type of file to load, can be 'csv' or 'excel'
            text_column (str): name of the column to load
            window (int): minimum length of the sentence to be accepted.
            regex_replace (Dict[str, str]): use regex to replace the keys of the dict with the values
            allowed_symbols (List[str]): a list of allowed symbols that won't be removed

        Raises:
            RuntimeError: allowed input data types are only 'csv' or 'excel'

        Returns:
            List[List[str]]: list containing the loaded sentences as word tokens
        """
        
        if file_type == 'csv':
            tweet_df = pd.read_csv(fname)
        elif file_type == 'excel':
            tweet_df = pd.read_excel(fname)
        else:
            raise RuntimeError('The only possible options for the input file type are \'csv\' or \'excel\'')
        
        texts = tweet_df[text_column]
        texts = texts.str.lower()
        texts = self._clean(texts, regex_replace, allowed_symbols)
        
        data = [] 
        
        for text in texts:
            for sentence in sent_tokenize(text):

                if sentence[-1] == '.':
                    sentence = sentence[:-1]
                
                tokens = sentence.split(' ')
                if len(tokens) < window + 1:
                    continue
                
                sentence_array = [word for word in tokens if len(word) > 0]
                data.append(sentence_array + ['.'])
        
        return data
    
    def _clean(self, texts_series: pd.Series, regex_replace: Dict[str, str],
               allowed_symbols: List[str]) -> pd.Series:
        """
        Clean the input sentences

        Args:
            texts_series (pd.Series): loaded sentences
            regex_replace (Dict[str, str]): use regex to replace the keys of the dict with the values
            allowed_symbols (List[str]): a list of allowed symbols that won't be removed

        Returns:
            pd.Series: loaded sentences after cleaning
        """
        texts_series = texts_series.apply(tc.remove_urls)
        texts_series = texts_series.apply(tc.remove_newlines)
        if len(regex_replace) > 0:
            texts_series = texts_series.apply(tc.replace_regex(regex_replace))
        texts_series = texts_series.apply(str.strip)
        if self._flatten_hashtags:
            texts_series = texts_series.apply(tc.flatten_hashtags)
        if self._flatten_mentions:
            texts_series = texts_series.apply(tc.flatten_mentions)
        texts_series = texts_series.apply(tc.clean_symbols(allowed_symbols))

        return texts_series