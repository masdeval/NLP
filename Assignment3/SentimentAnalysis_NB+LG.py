import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression

#A Logistic Regression Classifier using as parameter the log ratio of the MLE Naïve Bayes probabilities.

positive = open("./pos.tok").read()
negative = open("./neg.tok").read()

size_positive = len(positive.split("\n"))
size_negative = len(negative.split(("\n")))

target_pos = [1 for v in range(size_positive)]
target_neg = [0 for v in range(size_negative)]

target = target_pos + target_neg
data = positive.split("\n") + negative.split("\n")

############# First - BOW
bow = CountVectorizer(binary=True)
X = bow.fit_transform(data)
Y = target

X_matrix = X.toarray()
# Counts for class 1
p = np.sum(X_matrix[[v for v in range(size_positive)],:],axis=0)
p = p + 1
# Counts for class 0
q = np.sum(X_matrix[[v for v in range(size_positive,size_positive+size_negative)],:],axis=0)
q = q +1

r = np.log((p / np.sum(p))/(q / np.sum(q)))

X_new = X_matrix * r

# Logistic Regression with L2 regularization
result = cross_validate(LogisticRegression(penalty='l2'),X=X_new,y=Y,cv=10,scoring=['accuracy','f1'], return_train_score=False)

from prettytable import PrettyTable

print("\n Result for BOW features")
x = PrettyTable()
x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Fold 6", "Fold 7", "Fold 8", "Fold 9", "Fold 10"]
x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
print(x)
print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
print("Overall F1-score: %f" % np.mean(result['test_f1']))


############# Second - NGram
ngram = CountVectorizer(binary=True, ngram_range=(1,2))
X = ngram.fit_transform(data)
Y = target

X_matrix = X.toarray()
# Counts for class 1
p = np.sum(X_matrix[[v for v in range(size_positive)],:],axis=0)
p = p + 1
# Counts for class 0
q = np.sum(X_matrix[[v for v in range(size_positive,size_positive+size_negative)],:],axis=0)
q = q +1

r = np.log((p / np.sum(p))/(q / np.sum(q)))

X_new = X_matrix * r

# Logistic Regression with L2 regularization
result = cross_validate(LogisticRegression(penalty='l2'),X=X_new,y=Y,cv=10,scoring=['accuracy','f1'], return_train_score=False)

print("\n Result for NGram features")
x = PrettyTable()
x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Fold 6", "Fold 7", "Fold 8", "Fold 9", "Fold 10"]
x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
print(x)
print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
print("Overall F1-score: %f" % np.mean(result['test_f1']))
