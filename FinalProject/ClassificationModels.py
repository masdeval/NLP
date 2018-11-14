import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_validate
import pandas as pd
import os


def loadTrainData(path, data, target, limit = np.inf):
    #tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    tokenizer = gensim.utils.simple_preprocess
    i = 0
    if (path.__contains__('train_semeval')):
        for filename in os.listdir(path):
            with open(path+'/'+filename, encoding='latin-1') as f:
                try:
                    for line in f:
                        sentiment = line.split("\t")[1].strip()
                        if sentiment not in ('positive', 'negative', 'neutral'):
                          continue
                        if sentiment in ('neutral'):
                            continue
                        if sentiment in ('negative'):
                          sentiment = 0
                        elif sentiment in ('positive'):
                          sentiment = 1
                        target.append(sentiment)
                        tweet = line.split('\t')[2]
                        # data.append(tokenizer.tokenize(tweet))
                        data.append((tokenizer(tweet)))
                except:
                    continue
    else:
        with open(path, encoding='latin-1') as f:
            for line in f:
                i += 1
                if (i == limit):
                    return data, target
                line = line.replace('"', '')
                if (path.__contains__('hcr-train.csv')):
                    try:
                        sentiment = str(line.split(',')[4]).strip()
                        if sentiment not in ('positive','negative','irrelevant','neutral'):
                            continue
                        if sentiment in ('irrelevant','neutral'):
                            continue
                        if sentiment in  ('negative'):
                            sentiment = 0
                        elif sentiment in ('positive'):
                            sentiment = 1
                        target.append(sentiment)
                        tweet = line.split(',')[3]
                        #data.append(tokenizer.tokenize(tweet))
                        data.append((tokenizer(tweet)))
                    except Exception:
                        continue
                elif(path.__contains__('sanders-full-corpus.csv')):
                    try:
                        sentiment = line.split(',')[1].strip()
                        if sentiment not in ('positive','negative','irrelevant','neutral'):
                            continue
                        if sentiment in ('irrelevant','neutral'):
                            continue
                        if sentiment in  ('negative'):
                            sentiment = 0
                        elif sentiment in ('positive'):
                            sentiment = 1
                        target.append(sentiment)
                        tweet = line.split(',')[4]
                        #data.append(tokenizer.tokenize(tweet))
                        data.append((tokenizer(tweet)))
                    except Exception:
                        continue
                elif(path.__contains__('Tromp_en_UK_neg.csv')):
                    sentiment = 0
                    target.append(sentiment)
                    tweet = line
                    #data.append(tokenizer.tokenize(tweet))
                    data.append((tokenizer(tweet)))
                elif(path.__contains__('Tromp_en_UK_pos.csv')):
                    sentiment = 1
                    target.append(sentiment)
                    tweet = line
                    #data.append(tokenizer.tokenize(tweet))
                    data.append((tokenizer(tweet)))
    return data,target


def readWordvec(file, kv = True):
    if kv == True:
        return KeyedVectors.load(file, mmap='r')
    else:
        return Word2Vec.load(file)

def buildTwitterVector(tokens, word2vec, size=150):
    vec = np.zeros(size)
    count = 0.
    for word in tokens:
        try:
            vec += word2vec[word] #* tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not present
            continue
    if count != 0:
        vec /= count

    assert(len(vec) == size)
    return vec

def classification(classifier, features, target):
    # Training a classifier using the word vectors as features
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0, shuffle=True)
    classifier = classifier.fit(X=X_train,y=y_train)
    print(classifier.score(X_test,y_test))
    return classifier


data = []
sentiment = []
data,sentiment = loadTrainData("./training/hcr-train.csv", data, sentiment)
data,sentiment = loadTrainData("./training/sanders-full-corpus.csv", data, sentiment)
data,sentiment = loadTrainData("./training/Tromp_en_UK_neg.csv", data, sentiment)
data,sentiment = loadTrainData("./training/Tromp_en_UK_pos.csv", data, sentiment)
data,sentiment = loadTrainData("./training/train_semeval", data, sentiment)

features = []
model = readWordvec("./twitter_vectors_V4.kv", kv = True)
# Creating a representation for the whole tweet using wordvec

for tweet in data:
    features.append(buildTwitterVector(tweet,model))

result = cross_validate(LogisticRegression(penalty='l2'),X=features,y=sentiment,cv=5,scoring=['accuracy','f1'], return_train_score=False)

from prettytable import PrettyTable
print("\n Logistic in train")
x = PrettyTable()
x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
print(x)
print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
print("Overall F1-score: %f" % np.mean(result['test_f1']))

'''
print("\nLogistic in test - Apple-Twitter-Sentiment")
logistic = result['estimator'][0]
data = pd.read_csv("./specific_domain/Apple-Twitter-Sentiment-DFE.csv",delimiter=',',encoding='latin-1')
features = []
target = []
for index,row in data.iterrows():
    sentiment = row['sentiment']
    if sentiment not in ('1','2','3','4','5'):
        continue
    if sentiment in ('3'):
        continue
    if sentiment in ('1','2'):
        target.append(0)
    elif sentiment in ('4','5'):
        target.append(1)

    tweet = gensim.utils.simple_preprocess(row['text'])
    #tweet = TweetTokenizer(preserve_case=False, strip_handles=True).tokenize(row['text'])
    features.append(buildTwitterVector(tweet,model))

score = logistic.score(X=features, y=target)
print("Accuracy: %f" % score)

print("\nLogistic in test - Claritin Side Effects")
logistic = result['estimator'][0]
data = pd.read_csv("./specific_domain/claritin_october_twitter_side_effects.csv",delimiter=',',encoding='latin-1',dtype=str)
features = []
target = []
for index,row in data.iterrows():
    sentiment = row['sentiment']
    if sentiment not in ('1','2','3','4','5'):
        continue
    if sentiment in ('3'):
        continue
    if sentiment in ('1','2'):
        target.append(0)
    elif sentiment in ('4','5'):
        target.append(1)

    tweet = gensim.utils.simple_preprocess(row['content'])
    #tweet = TweetTokenizer(preserve_case=False, strip_handles=True).tokenize(row['content'])
    features.append(buildTwitterVector(tweet,model))

score = logistic.score(X=features, y=target)
print("Accuracy: %f" % score)

print("\nLogistic in test - Twitter Sentiment Self-drive-car")
logistic = result['estimator'][0]
data = pd.read_csv("./specific_domain/Twitter-sentiment-self-drive-DFE.csv",delimiter=',',encoding='latin-1',dtype=str)
features = []
target = []
for index,row in data.iterrows():
    sentiment = row['sentiment']
    if sentiment not in ('1','2','3','4','5'):
        continue
    if sentiment in ('3'):
        continue
    if sentiment in ('1','2'):
        target.append(0)
    elif sentiment in ('4','5'):
        target.append(1)

    tweet = gensim.utils.simple_preprocess(row['text'])
    #tweet = TweetTokenizer(preserve_case=False, strip_handles=True).tokenize(row['text'])
    features.append(buildTwitterVector(tweet,model))

score = logistic.score(X=features, y=target)
print("Accuracy: %f" % score)

print("\nLogistic in test - US Airlines")
logistic = result['estimator'][0]
data = pd.read_csv("./specific_domain/AirlineSentiment.csv",delimiter=',',encoding='latin-1',dtype=str)
features = []
target = []
for index,row in data.iterrows():
    sentiment = row['airline_sentiment']
    if sentiment not in ('positive','negative','neutral'):
        continue
    if sentiment in ('neutral'):
        continue
    if sentiment in ('negative'):
        target.append(0)
    elif sentiment in ('positive'):
        target.append(1)

    tweet = gensim.utils.simple_preprocess(row['text'])
    #tweet = TweetTokenizer(preserve_case=False, strip_handles=True).tokenize(row['text'])
    features.append(buildTwitterVector(tweet,model))

score = logistic.score(X=features, y=target)
print("Accuracy: %f" % score)
'''