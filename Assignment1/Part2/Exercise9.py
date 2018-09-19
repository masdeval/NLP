# Language Models
# Language identification is the problem of taking a text in an unknown language and
# determining what language it is written in. N-gram models provide a very effective
# solution for this problem. For training, use the English (EN.txt), French (FR.txt), and
# German (GR.txt) texts made available in blackboard/piazza. For test, use the file
# LangID.test.txt. For each of the following problems, the output of your program
# has to contain a list of [line_id] [language] pairs:
# ID LANG
# 1   EN
# 2   FR
# 3   GR

#Implement a word bigram model, which learns word bigram
#probabilities from the training data. Again, a separate model will be learned for
#each language. Use Good Turing smoothing to avoid zero-counts in the data.
#Apply the models to determine the language ID for each sentence in the test file


from nltk import bigrams
from collections import defaultdict, Counter
import re

def makeGoodTuringModel(fileName, testName):

 # reading the train set
 sentences = open(fileName).read()
 #lowering case
 sentences = sentences.lower()
 #removing punctuation
 sentences = re.sub(r"\W", ' ', sentences)
 #replace sequences of blank characteres to a single one
 sentences = re.sub(r"\s+", ' ', sentences)
 vocabularyTrain = set(sentences.split())

 # reading the test set
 testSentences = open(testName).read()
 testSentences = testSentences.lower()
 testSentences = re.sub(r"\W", ' ', testSentences)
 testSentences = re.sub(r"\s+", ' ', testSentences)
 testSentences = re.sub(r'^\d+\s', '', testSentences)  # remove the initial number from each row
 vocabularyTest = set(testSentences.split())

 vocabulary = vocabularyTrain.union(vocabularyTest)

 # bigrams in train
 vocabularyCount = Counter(bigrams(sentences.split()))

 # generating the V² entries and setting singleton to 0
 # for w1 in vocabulary:
 #     for w2 in vocabulary:
 #         if (not vocabularyCount.__contains__((w1,w2))):
 #             vocabularyCount[(w1,w2)] = 0
 # this way getting Memory Error
 N_zero = 0
 for w1 in vocabulary:
      for w2 in vocabulary:
          if (not vocabularyCount.__contains__((w1,w2))):
              N_zero = N_zero + 1

 # inversing the counting we get N
 N = Counter(vocabularyCount.values())

 struct = dict()
 struct['vocabulary_size'] = len(vocabulary)
 struct['N'] = N
 struct['count'] = vocabularyCount
 struct['N_zero'] = N_zero

 return struct

def getZeroCountProbability(model):
    return (model['N'].get(1)/model['vocabulary_size'])/model['N_zero']

def getProbability(model,bigram,k=5):

    if (model['count'].get(bigram) > k):
        return model['count'].get(bigram) / model['vocabulary_size']

    numerator =  (model['count'].get(bigram)+1) * (model['N'].get(model['count'].get(bigram)+1)/model['N'].get(model['count'].get(bigram))) - (model['count'].get(bigram) * ((k+1)*model['N'].get(k+1))/model['N'].get(1))
    denominator = 1 - ((k+1)*model['N'].get(k+1)/model['N'].get(1))
    estimatedDiscount = numerator / denominator

    return estimatedDiscount / model['vocabulary_size']


import math
test_file = open('./LangID.test.txt')
test_results = defaultdict(lambda : defaultdict( lambda : float))
modelEN = makeGoodTuringModel('./EN.txt','LangID.test.txt')
modelFR = makeGoodTuringModel('./FR.txt','LangID.test.txt')
modelGR = makeGoodTuringModel('./GR.txt','LangID.test.txt')
result = list()

for i,line in enumerate(test_file):
    line = line.lower()
    line = re.sub(r"\W", ' ', line)
    line = re.sub(r"\s+", ' ', line)
    line = re.sub(r'^\d+\s','',line) # remove the initial number from each row

    #EN
    probability = 0.0
    for w1,w2 in bigrams(line.split()):
        if(not modelEN['count'].__contains__((w1,w2))):
            probability += math.log(getZeroCountProbability(modelEN))
        else:
            probability += math.log(getProbability(modelEN,(w1,w2)))

    test_results[i]['EN'] = probability

    # FR
    probability = 0.0
    for w1, w2 in bigrams(line.split()):
        if (not modelFR['count'].__contains__((w1, w2))):
            probability += math.log(getZeroCountProbability(modelFR))
        else:
            probability += math.log(getProbability(modelFR, (w1, w2)))

    test_results[i]['FR'] = probability

    # GR
    probability = 0.0
    for w1, w2 in bigrams(line.split()):
        if (not modelGR['count'].__contains__((w1, w2))):
            probability += math.log(getZeroCountProbability(modelGR))
        else:
            probability += math.log(getProbability(modelGR,(w1,w2)))

    test_results[i]['GR'] = probability

    result.append(max(test_results[i], key=test_results[i].get))
    print(str(i+1) + " " + max(test_results[i], key=test_results[i].get))


hit = 0
i = 0
testFile = open('./LangID.gold.txt')
for line in testFile:
    ss = line.split()
    if (ss[1] == 'EN' or ss[1] == 'GR' or ss[1] == 'FR'):
        if(result[i] == ss[1]):
            hit = hit + 1
        i = i + 1 #if the line is about the result, add one


print("Accuracy is: " + str(hit/i))
testFile.close()
