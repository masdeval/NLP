import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_validate
import pandas as pd
import os


def loadTrainData(path, data, target, limit = np.inf):
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
                        data.append(((tweet)))
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
                        data.append(((tweet)))
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
                        data.append(((tweet)))
                    except Exception:
                        continue
                elif(path.__contains__('Tromp_en_UK_neg.csv')):
                    sentiment = 0
                    target.append(sentiment)
                    tweet = line
                    #data.append(tokenizer.tokenize(tweet))
                    data.append(((tweet)))
                elif(path.__contains__('Tromp_en_UK_pos.csv')):
                    sentiment = 1
                    target.append(sentiment)
                    tweet = line
                    #data.append(tokenizer.tokenize(tweet))
                    data.append(((tweet)))
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
            vec += word2vec[word]
            count += 1.
        except KeyError: # handling the case where the token is not present
            continue
    if count != 0:
        vec /= count

    assert(len(vec) == size)
    return vec

def buildTwitterVectorTFIDF(tokens, word2vec, tfidfVectorizer, tfidf, size=150):
    vec = np.zeros(size)
    count = 0.
    for word in tokens:
        try:
            vec += word2vec[word] * tfidf[0,tfidfVectorizer.vocabulary_[word]]
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
model = KeyedVectors.load_word2vec_format("./glove.twitter.27B/word2vec200d.txt", binary=False)

######  TF-IDF #####
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(encoding='latin-1', vocabulary=model.wv.vocab.keys(),lowercase=True)
tfidf = tfidfVectorizer.fit_transform(data)
#####################

# Creating a representation for the whole tweet using Glove wordvec
import preprocess_twitter as stanfordPreprocessing
for i,tweet in enumerate(data):

    tweet = stanfordPreprocessing.tokenize(tweet).split()
    #Without TF_IDF
    #features.append(buildTwitterVector(tweet,model,size=200))

    #With TF_IDF - do not remove punctuation since Glove was trained with it
    features.append(buildTwitterVectorTFIDF(tweet, model, tfidfVectorizer, tfidf.getrow(i).toarray(), size=200))


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

result = cross_validate(svm.SVC(C=1.0,kernel='linear'),X=features,y=sentiment,cv=5,scoring=['accuracy','f1'], return_train_score=False)

from prettytable import PrettyTable
print("\n SVM in train")
x = PrettyTable()
x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
print(x)
print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
print("Overall F1-score: %f" % np.mean(result['test_f1']))



#Saving SVM classifier
#classifier = classification(svm.SVC(C=1.0,kernel='linear'),features,sentiment)
#from joblib import dump
#dump(classifier, 'svm_model.save')
