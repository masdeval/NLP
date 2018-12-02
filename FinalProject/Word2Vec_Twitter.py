import numpy as np
from string import punctuation
from random import shuffle
from gensim.test.utils import get_tmpfile
import gensim
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import time
from nltk.tokenize import TweetTokenizer


def load1_6million(path, tokenizer = gensim.utils.simple_preprocess, limit = np.inf):
    #tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    data = []
    target = []
    i = 0
    with open(path, encoding='latin-1') as f:
        for line in f:
            i += 1
            line = line.replace('"', '')
            sentiment = line.split(',')[1]
            if sentiment not in ('0','1','2','3','4'):
                continue
            if sentiment in  ('0'):
                sentiment = 0
            elif sentiment in ('4'):
                sentiment = 1
            target.append(sentiment)
            tweet = line.split(',')[-1]
            data.append(tokenizer(tweet))
            #data.append(tokenizer.tokenize(tweet))
            if (i == limit):
                return data, target
    return data,target


def saveWordvec(wordvec):
    kv = './twitter_vectors.kv'
    model = "./twitter_vectors.model"
    #fname = get_tmpfile("./twitter_vectors.kv") # if we want to save the file in the default temporary system directory
    wordvec.wv.save(kv) #save the KeyedVectors
    #fname = get_tmpfile("./twitter_vectors.model")
    wordvec.save(model) #save the whole model

def readWordvec(file, kv = True):
    if kv == True:
        return KeyedVectors.load(file, mmap='r')
    else:
        return Word2Vec.load(file)


# Read each line of the text file and return a list of tokens for each tweet and the associated sentiment
data,sentiment = load1_6million ("./kaggle_sentiment140/random_training.csv", tokenizer=gensim.utils.simple_preprocess)

# Creating the custom embedings for our specific domain
begin = time.perf_counter()
# SkipGram = True
model = gensim.models.Word2Vec(data, size=200, window=5, min_count=5, workers=10, sg=1)
model.train(data, total_examples=len(data), epochs=10)
end = time.perf_counter()
print("\nTime elapsed: " + str((end-begin)/60) + " min")
saveWordvec(model)

#model = readWordvec("./twitter_vectors.kv_V1", kv = True)
#print(model.most_similar('good'))


### Augumenting the word vectors
# Read each line of the text file and return a list of tokens for each tweet and the associated sentiment
#nltk.tokenize.TweetTokenizer # a tweet tokenizer from nltk
#data,sentiment = loadTwitterKaggle("./kaggle_Twitter_sentiment_analysis/train.csv", tokenizer=gensim.utils.simple_preprocess)
# # Retrain word2vec using new dataset
# model = readWordvec("./twitter_vectors.model_V1", kv = False)
# # Creating the custom embedings for our specific domain
# begin = time.perf_counter()
# # adding more vocabulary into the previous one
# model.build_vocab(data, update=True)
# model.train(data, total_examples=len(data), epochs=model.epochs)
# end = time.perf_counter()
# print("\nTime elapsed: " + str((end-begin)/60) + " min")
# saveWordvec(model)


