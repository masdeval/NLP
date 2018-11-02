import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import  LogisticRegression

#Logistic Regression classifier with L2 regularization using binary bag-of-words and n-gram features.

positive = open("./pos.tok").read()
negative = open("./neg.tok").read()

size_positive = len(positive.split("\n"))
size_negative = len(negative.split(("\n")))

target_pos = [1 for v in range(size_positive)]
target_neg = [0 for v in range(size_negative)]

target = target_pos + target_neg
data = positive.split("\n") + negative.split("\n")

###### First - BOW
bow = CountVectorizer(binary=True)
X = bow.fit_transform(data)
Y = target

result = cross_validate(LogisticRegression(penalty='l2'),X=X,y=Y,cv=10,scoring=['accuracy','f1'], return_train_score=False)

from prettytable import PrettyTable
print("\n BOW features")
x = PrettyTable()
x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Fold 6", "Fold 7", "Fold 8", "Fold 9", "Fold 10"]
x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
print(x)
print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
print("Overall F1-score: %f" % np.mean(result['test_f1']))


###### Second - N-Gram
ngram = CountVectorizer(binary=True,ngram_range=(1,2))
X = ngram.fit_transform(data)
Y = target

result = cross_validate(LogisticRegression(penalty='l2'),X=X,y=Y,cv=10,scoring=['accuracy','f1'], return_train_score=False)

print("\n N-Gram features")
x = PrettyTable()
x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Fold 6", "Fold 7", "Fold 8", "Fold 9", "Fold 10"]
x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
print(x)
print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
print("Overall F1-score: %f" % np.mean(result['test_f1']))
