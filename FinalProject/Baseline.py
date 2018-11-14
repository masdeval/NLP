import numpy as np
import gensim
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from nltk.tokenize import TweetTokenizer
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
                        data.append(' '.join(tokenizer(tweet)))
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
                        data.append(' '.join(tokenizer(tweet)))
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
                        data.append(' '.join(tokenizer(tweet)))
                    except Exception:
                        continue
                elif(path.__contains__('Tromp_en_UK_neg.csv')):
                    sentiment = 0
                    target.append(sentiment)
                    tweet = line
                    #data.append(tokenizer.tokenize(tweet))
                    data.append(' '.join(tokenizer(tweet)))
                elif(path.__contains__('Tromp_en_UK_pos.csv')):
                    sentiment = 1
                    target.append(sentiment)
                    tweet = line
                    #data.append(tokenizer.tokenize(tweet))
                    data.append(' '.join(tokenizer(tweet)))
    return data,target


data = []
sentiment = []
data,sentiment = loadTrainData("./training/hcr-train.csv", data, sentiment)
data,sentiment = loadTrainData("./training/sanders-full-corpus.csv", data, sentiment)
data,sentiment = loadTrainData("./training/Tromp_en_UK_neg.csv", data, sentiment)
data,sentiment = loadTrainData("./training/Tromp_en_UK_pos.csv", data, sentiment)
data,sentiment = loadTrainData("./training/train_semeval", data, sentiment)

ngram = CountVectorizer(lowercase=False, binary=True, ngram_range=(1,2))
X = ngram.fit_transform(data)
Y = sentiment

result = cross_validate(LogisticRegression(penalty='l2'),X=X,y=Y,cv=5,scoring=['accuracy','f1'], return_train_score=False)

from prettytable import PrettyTable
print("\n Logistic in train")
x = PrettyTable()
x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
print(x)
print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
print("Overall F1-score: %f" % np.mean(result['test_f1']))

