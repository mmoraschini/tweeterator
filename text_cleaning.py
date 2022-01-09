from typing import List, Callable
import re

def remove_urls(string: str) -> str:
    """
    Remove URLs from the text

    Args:
        string (str): input string

    Returns:
        str: input string with URLs removed (if any)
    """
    return re.sub(r'https?:\/\/.[^ ]* * ', 'website ', string)

def remove_newlines(string: str) -> str:
    """
    Remove newline characters from string

    Args:
        string (str): input string

    Returns:
        str: input string with newlines removed
    """
    string = string.replace('\n', ' ')
    return string.replace('. .', '')

def remove_words(regex_to_remove: List[str]) -> Callable:
    """
    Remove words from input string based on regulr expressions

    Args:
        regex_to_remove (List[str]): list of regular expressions to apply

    Returns:
        Callable: a function that applies the regular expressions to a string
    """
    def run(string: str) -> str:
        """

        Args:
            string (str): the input string

        Returns:
            str: the input string with words removed
        """
        for regex in regex_to_remove:
            string = re.sub(regex, '', string)
        
        return string
    
    return run

def clean_symbols(string: str) -> str:
    """
    Remove symbols from input string apart from word characters, points, spaces, hashtags and at symbols

    Args:
        string (str): the input string

    Returns:
        str: the input string with symbols removed
    """
    string = string.replace("’", "'")
    string = string.replace("&amp;", "and")
    return re.sub(r'[^\w .\'#@]', '', string)

def flatten_mentions(string: str) -> str:
    """
    Substitute all mentions with @mention

    Args:
        string (str): the input string

    Returns:
        str: the input string with flattened mentions
    """
    return re.sub(r'(@+[a-zA-Z0-9(_)]{1,})', '@mention', string)

def flatten_hashtags(string: str) -> str:
    """
    Substitute all hashtags with #hashtag

    Args:
        string (str): the input string

    Returns:
        str: the input string with flattened hashtags
    """
    return re.sub("(#+[a-zA-Z0-9(_)]{1,})", "#hashtag", string)