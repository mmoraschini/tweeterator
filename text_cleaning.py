import re

def remove_urls(string):
    return re.sub(r'https?:\/\/.[^ ]* * ', 'website ', string)

def remove_newlines(string):
    string = string.replace('\n', ' ')
    return string.replace('. .', '')

def remove_words(regex_to_remove):
    def run(string):
        for regex in regex_to_remove:
            string = re.sub(regex, '', string)
        
        return string
    
    return run

def clean_symbols(string):
    string = string.replace("’", "'")
    return re.sub(r'[^\w .\'#@]', '', string)

def flatten_mentions(string):
    return re.sub(r'(@+[a-zA-Z0-9(_)]{1,})', '@mention', string)

def flatten_hashtags(string):
    return re.sub("(#+[a-zA-Z0-9(_)]{1,})", "#hashtag", string)