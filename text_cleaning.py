from typing import List, Dict, Callable
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

def replace_regex(regex: Dict[str, str]) -> Callable:
    """
    Remove words from input string based on regulr expressions

    Args:
        regex (List[str]): list of regular expressions to apply

    Returns:
        Callable: a function that applies the regular expressions to a string
    """
    def run(string: str) -> str:
        """

        Args:
            string (str): the input string

        Returns:
            str: the input string with words replaced
        """
        for k in regex:
            repl = regex[k]
            string = re.sub(k, repl, string)
        
        return string
    
    return run

def clean_symbols(string: str) -> str:
    """
    Remove symbols from input string apart from word characters, points, quotes, spaces, hashtags and at symbols

    Args:
        string (str): the input string

    Returns:
        str: the input string with symbols removed
    """
    
    return re.sub(r'[^\w .\'\"#@]', '', string)

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