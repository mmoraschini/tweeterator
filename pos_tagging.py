from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
import nltk


nltk.download('averaged_perceptron_tagger')


def get_tags(data: List[List[str]]) -> List[List[List[str]]]:
    """
    Convert a list of sentences, as list of list of words, to a list of lists of two-dimentional lists, the first element
    containing the word and the second the associated POS tag.

    Args:
        data (List[List[str]]): the input list of lists of words

    Returns:
        List[List[List[str]]]: the output list of lists of two-dimentional lists, the first element containing the word and the second the associated POS tag.
    """
    data_plus_pos = []
    for sentence in tqdm(data):
        words_and_pos_tags = nltk.pos_tag(sentence)
        words_and_pos_tags = [list(pos) for pos in words_and_pos_tags]
        data_plus_pos.append(words_and_pos_tags)
    data_plus_pos = np.array(data_plus_pos, dtype=object)

    return data_plus_pos


def get_frequency(data: List[List[str]], window: int) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], float]:
    """
    Get the frequency of the next POS tag given the previous `window` ones.

    Args:
        data (List[List[str]]): the input list of lists of words
        window (int): previous words sequence length

    Returns:
        Tuple[Dict[str, Dict[str, int]], Dict[str, int], float]:
            A dictionary mapping the string representation of the previous tags into the frequency of each observed nect tag
            A dictionary mapping the string representation of the previous tags into ints count
            The minimum observed frequency
    """
    dict_pos_freq = {}
    dict_pos_count = {}

    for sentence in tqdm(data):
        words_and_pos_tags = nltk.pos_tag(sentence)
        pos_tags = list(zip(*words_and_pos_tags))[1]
        for i in range(len(sentence) - window):
            preceding = str(pos_tags[i:i+window])
            following = pos_tags[i+window]
            try:
                dict_pos_count[preceding] += 1
                try:
                    # Increase the frequency of seeing the window preceding POSs and then the following POS
                    dict_pos_freq[preceding][following] += 1
                except KeyError:
                    dict_pos_freq[preceding][following] = 1
            except KeyError:
                dict_pos_freq[preceding] = {following: 1}
                dict_pos_count[preceding] = 1

    # Normalise each count to obtain a frequency
    for k1 in dict_pos_freq.keys():
        v1 = dict_pos_freq[k1]
        count = dict_pos_count[k1]
        for k2 in v1.keys():
            v1[k2] = v1[k2] / count
    
    min_freq = 1
    for k1 in dict_pos_freq.keys():
        v1 = dict_pos_freq[k1]
        for k2 in v1.keys():
            if v1[k2] < min_freq:
                min_freq = v1[k2]
    
    return dict_pos_freq, dict_pos_count, min_freq