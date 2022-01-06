import re

def remove_urls(string):
    return re.sub(r'https?:\/\/.[^ ]* * ', 'website ', string)

def remove_newlines(string):
    string = string.replace('\n', ' ')
    return string.replace('. .', '')

def remove_words(words_to_remove):
    def run(string):
        for word in words_to_remove:
            string = re.sub(word, '', string)
        
        return string
    
    return run

def clean_symbols(string, encoding='utf-8'):
    string = string.replace("’", "'")
    return re.sub(r'[^\w .\'#@]', '', string)

def flatten_mentions(string):
    return re.sub(r'(@+[a-zA-Z0-9(_)]{1,})', '@mention', string)

def flatten_hashtags(string):
    return re.sub("(#+[a-zA-Z0-9(_)]{1,})", "#hashtag", string)