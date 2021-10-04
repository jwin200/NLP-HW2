import re
import os
import nltk
import json
import multiprocessing
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


Q = None
POOL = None


def main():
    global POOL, Q
    labeled_reviews = []
    neg_path = './Homework2-Data/neg/'
    pos_path = './Homework2-Data/pos/'
    total_length = int(len(os.listdir(neg_path))) + int(len(os.listdir(pos_path)))
    print(f'Processing {total_length} files...')
    # Gather all reviews into a list
    for f in os.listdir(neg_path):
        with open(neg_path + f, 'rb') as file:
            review = str(file.read().lower())
            labeled_reviews.append((review, 'neg'))
    for f in os.listdir(pos_path):
        with open(pos_path + f, 'rb') as file:
            review = str(file.read().lower())
            labeled_reviews.append((review, 'pos'))

    # Tokenizing and data-cleaning
    i = 0
    data = []
    tokens = []
    start = datetime.now()
    p = re.compile(r'[^a-z]')
    stops = set(stopwords.words('english'))
    for words in labeled_reviews:
        for word in word_tokenize(words[0]):
            word = word.replace('\\n', '')
            word = word.replace('\\', '')
            word = word.replace('b\'', '')
            word = re.sub(r'[0-9]+', '', word)
            word = re.sub(r'[xx]+', '', word)
            word = word.strip('().-/!;:@#$^?*`')

            if len(word) > 3 and p.match(word) is None:
                if word not in tokens and word not in stops:
                    tokens.append(word)
        i += 1
        stats(len(labeled_reviews), start, i, 'Reading tokens...    ')
    tokens = clean_tokens(tokens)

    i = 0
    start = datetime.now()
    for x in labeled_reviews:
        POOL.apply(aggregate_data, (tokens, x[0], x[1],))
        i += 1
        stats(len(labeled_reviews), start, i, 'Creating processes...     ')

    i = 0
    start = datetime.now()
    while True:
        output = Q.get()
        if output is not None:
            data.append(output)
            i += 1
            stats(len(labeled_reviews), start, i, 'Finishing processes...      ')
        if i % 50 == 0:
            with open('data.json', 'w') as save_file:
                json.dump({'data': data}, save_file, indent=4)
        if i == len(labeled_reviews):
            with open('data.json', 'w') as save_file:
                json.dump({'data': data}, save_file, indent=4)
            break
    print(f'Data has been generated and saved')


def aggregate_data(tokens, document, sentiment):
    """Given a document and all possible tokens, return a
       dict of each token and whether it appears in the document."""
    global Q
    d = {}
    for word in tokens:
        d.update({word: (word in document)})
        # if word in document:
        #     print(f'{word}: \n{document}')
    t = (d, sentiment)
    Q.put(t)


def clean_tokens(tokens):
    """If two words are split by '/' or '.' within a token, divide words
       and return updated token list. Additionally, lemmatize all tokens."""
    new_tokens = []
    # Separate conjoined words
    for word in tokens:
        if '.' in word:
            new_word1 = re.split('.', word)[0]
            new_word2 = re.split('.', word)[1]
            if len(new_word1) > 3:
                new_tokens.append(new_word1)
            if len(new_word2) > 3:
                new_tokens.append(new_word2)
        elif '/' in word:
            new_word1 = re.split('/', word)[0]
            new_word2 = re.split('/', word)[1]
            if len(new_word1) > 3:
                new_tokens.append(new_word1)
            if len(new_word2) > 3:
                new_tokens.append(new_word2)
        else:
            new_tokens.append(word)

    # Lemmatize common suffixes
    p = re.compile(r'.*ly$')
    q = re.compile(r'.*s$')
    for word in new_tokens:
        if p.match(word):
            new_word = word[:(len(word)-2)]
            if new_word in new_tokens:
                new_tokens.remove(word)
        if q.match(word):
            new_word = word[:(len(word)-1)]
            if new_word in new_tokens:
                new_tokens.remove(word)
        if len(word) < 4:
            new_tokens.remove(word)
    return new_tokens


def stats(length, start, i, message):
    """Display process statistics."""
    ave_time = (datetime.now() - start).seconds / i
    minutes_left = int(((length - i) * ave_time) / 60)
    if minutes_left > 0:
        minutes_message = f'Approximately {minutes_left} minute(s) remaining'
    else:
        minutes_message = f'Less than a minute remaining                    '

    print(f'{message}\n'
          f'{round((i / length) * 100, 2)}% done     \n'
          f'{minutes_message}',
          end='\r\033[A\r\033[A\r')


def setup():
    """Initialize starting values and global variables."""
    global Q, POOL
    multiprocessing.set_start_method('fork')
    Q = multiprocessing.Manager().Queue()
    POOL = multiprocessing.Pool(6)
    nltk.download('punkt')
    nltk.download('stopwords')


if __name__ == '__main__':
    setup()
    main()
