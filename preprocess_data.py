import pandas as pd
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)

from string import punctuation
from itertools import dropwhile

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer
tweettokenizer = TweetTokenizer()
from nltk.corpus import stopwords
from nltk.stem import *

from gensim.models.word2vec import Word2Vec 
from gensim.models import KeyedVectors

from sklearn.preprocessing import scale
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding

def clean_text(tweet):
    """remove punctuation, stopword and other unimportant word  of a string

    :param text: string
    """
    tweet = tweet.lower()
    tweet = ''.join([c for c in tweet if c not in punctuation])

    tweet = tweettokenizer.tokenize(tweet)
    tweet = filter(lambda t: not t.startswith('@'), tweet)
    tweet = filter(lambda t: not t.startswith('#'), tweet)
    tweet = list(filter(lambda t: not t.startswith('http'), tweet))
    stops = set(stopwords.words("english"))
    tweet = [w for w in tweet if not w in stops and len(w) >= 3]
    
    stemmer = SnowballStemmer('english')
    tweet = [stemmer.stem(word) for word in tweet]
    return tweet

def preprocess_data(dataframe):
    """ Prepare tweet dataframe

    :param dataframe:pandas dataframe
    """
    data = dataframe[['labels','text']].copy()
    data['sentiment'] = data['labels'].progress_map(lambda x: 1 if int(x) > 0
            else 0).copy()
    logging.info('Encoded Labels')
    data['tokens'] = data['text'].progress_map(clean_text).copy()
    logging.info('Cleaned Tweets')
    return data

def word2vec_build(corpus, dim = 100, min_count = 10):
    """
    build word2vec model from corpus
    """
    model = Word2Vec(size=dim, min_count= min_count)
    model.build_vocab([x for x in tqdm(corpus)])
    model.train([x for x in tqdm(corpus)], total_examples=model.corpus_count, 
            epochs=model.iter)
    return model

def word2vec_generate_helper(tokens_sentence, model, size):
    temp = np.zeros(size).reshape((1, size)) 
    count = 0
    for word in tokens_sentence:
        try:
            temp += model[word].reshape((1, size)) #* tfidf[word]
            count += 1.
        except KeyError: # ignore the case where word is not in w2v
            continue

    if count != 0:
        temp /= count
    return temp

def word2vec_generate(tokens_sentence_list, model, dim = 100):
    """
    output word2vec of sentence_list based on model
    """
    vectors = np.concatenate([word2vec_generate_helper(x, model, dim) for
        x in tqdm(tokens_sentence_list)])
    vectors = scale(vectors)
    logging.info("word2vec generated")
    return vectors

def create_sequence(tweet, vocabulary_size = 20000):
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(tweet)
    sequences = tokenizer.texts_to_sequences(tweet)
    data = pad_sequences(sequences, maxlen=50)
    return data

def single_layer_lstm_w2v(x_train, y_train, x_test, y_test):
    """
    This function preprocesses with word2vec and train with LSTM
    """
    x_train = np.expand_dims(x_train, axis = 2) 
    x_test = np.expand_dims(x_test, axis = 2) 
    
    model = Sequential()
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    logging.info('Test Accuracy {}%'.format(score[1]*100))
    return model

def bidirectional_lstm_w2v(x_train, y_train, x_test, y_test):
    """
    This function preprocesses with word2vec and train with BiLSTM
    """
    x_train = np.expand_dims(x_train, axis = 2) 
    x_test = np.expand_dims(x_test, axis = 2) 
    model = Sequential()
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    logging.info('Test Accuracy {}%'.format(score[1]*100))
    return model

def single_layer_lstm_embedding(x_train, y_train, x_test, y_test):
    """
    This function preprocesses with embedding and train with LSTM
    :param x_train: train_vecs from embedding
    """
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=50))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    logging.info('Test Accuracy {}%'.format(score[1]*100))
    return model

def bidirectional_lstm_embedding(x_train, y_train, x_test, y_test):
    """
    This function preprocesses with embedding and train with BiLSTM
    :param x_train: train_vecs from embedding
    """
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=50))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    logging.info('Test Accuracy {}%'.format(score[1]*100))
    return model


