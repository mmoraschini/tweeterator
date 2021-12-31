import re

def remove_urls(string):
    return re.sub(r'https?:\/\/.[^ ]* *', 'website', string)

def remove_newlines(string):
    string = string.replace('\n', ' ')
    return string.replace('. .', '')

def remove_words(words_to_remove):
    def run(string):
        for word in words_to_remove:
            string = string.replace(word, '')
        
        return string
    
    return run

def clean_symbols(string, encoding='utf-8'):
    string = string.replace("’", "'")
    return re.sub(r'[^\w .\']', '', string)
#    return bytes(string, encoding).decode(encoding, 'ignore')
#    return string.decode(encoding, 'ignore').encode(encoding)

def flatten_mentions(string):
    return re.sub(r'@[^ ]* *', 'personsname', string)