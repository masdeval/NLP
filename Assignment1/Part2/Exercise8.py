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

# Implement a word bigram model, which learns word bigram
# probabilities from the training data. Again, a separate model will be learned for
# each language. Use Add-One smoothing to avoid zero-counts in the data. Apply
# the models to determine the language ID for each sentence in the test file.

from nltk import bigrams
from collections import defaultdict, Counter
import re

def makeSmoothModel(fileName):

 sentences = open(fileName).read()
 #lowering case
 sentences = sentences.lower()
 #removing punctuation
 sentences = re.sub(r"\W", ' ', sentences)
 #replace sequences of blank characteres to a single one
 sentences = re.sub(r"\s+", ' ', sentences)

 # model[w1][w2] stores the number of times each bigram w1,w2 was seen
 # Add-One Smoothing by initializing with 1
 modelCount = defaultdict(lambda : defaultdict(lambda:1))
 modelProbability = defaultdict(lambda: defaultdict(lambda: 0.0))

 # counting bigrams
 bigram = bigrams(sentences.split())
 # counting the number of specific bigram pairs
 for w1, w2 in bigram:
   modelCount[w1][w2] += 1
   assert modelCount[w1][w2] >= 2

 vocabulary = Counter(bigrams(sentences.split())) #number of different bigrams
 # getting probabilities
 for w1 in modelCount:
    # Smoothing
    # Number of times a bigram begining with w1 has ocured + the size of the model vocabulary
    total_count = (sum(modelCount[w1].values())) + len(vocabulary) - len(modelCount[w1].keys())# counting with the +1 for each bigram = (N+V)
    for w2 in modelCount[w1]:
        modelProbability[w1][w2] = modelCount[w1][w2]/total_count # calculating the relative frequency for each bigram
        assert(modelProbability[w1][w2] !=0 )

 return modelCount, modelProbability


def updateModel(modelCount,modelProbability,w1,w2):

    modelCount[w1][w2] = 1
    vocabulary = set({(w1,w2) for w1 in modelCount.keys() for w2 in modelCount[w1].keys()})
    for w1 in modelCount:
       total_count = (sum(modelCount[w1].values())) + len(vocabulary) - len(modelCount[w1].keys()) #
       for w2 in modelCount[w1]:
           modelProbability[w1][w2] = modelCount[w1][w2] / total_count  # calculating the relative frequency for each bigram
           assert (modelProbability[w1][w2] != 0)
    return modelCount, modelProbability

import math
test_file = open('./LangID.test.txt')

test_results = defaultdict(lambda : defaultdict( lambda : float))

modelCountEN, modelProbabilityEN = makeSmoothModel('./EN.txt')
modelCountFR, modelProbabilityFR = makeSmoothModel('./FR.txt')
modelCountGR, modelProbabilityGR = makeSmoothModel('./GR.txt')

result = list()

for i,line in enumerate(test_file):
    line = line.lower()
    line = re.sub(r"\W", ' ', line)
    line = re.sub(r"\s+", ' ', line)
    line = re.sub(r'^\d+\s','',line) # remove the initial number from each row

    #EN
    probability = 0.0
    # Create the model from th train set
    # Verify whether the bigrams in the test are already in the train
    # If not, add it and recalculate the probabilities
    for w1,w2 in bigrams(line.split()):
        if(w1 not in modelCountEN.keys()):
            modelCountEN, modelProbabilityEN = updateModel(modelCountEN, modelProbabilityEN, w1, w2)
        elif (w2 not in modelCountEN[w1].keys()):
            modelCountEN, modelProbabilityEN = updateModel(modelCountEN, modelProbabilityEN, w1, w2)

    for w1,w2 in bigrams(line.split()):
        probability += math.log(modelProbabilityEN[w1][w2])

    test_results[i]['EN'] = probability

    # FR
    probability = 0.0
    for w1, w2 in bigrams(line.split()):
        if(w1 not in modelCountFR.keys()):
            modelCountFR, modelProbabilityFR = updateModel(modelCountFR, modelProbabilityFR, w1, w2)
        elif (w2 not in modelCountFR[w1].keys()):
            modelCountFR, modelProbabilityFR = updateModel(modelCountFR, modelProbabilityFR, w1, w2)

    for w1, w2 in bigrams(line.split()):
        probability += math.log(modelProbabilityFR[w1][w2])

    test_results[i]['FR'] = probability

    # GR
    probability = 0.0
    for w1, w2 in bigrams(line.split()):
        if(w1 not in modelCountGR.keys()):
            modelCountGR, modelProbabilityGR = updateModel(modelCountGR, modelProbabilityGR, w1, w2)
        elif (w2 not in modelCountGR[w1].keys()):
            modelCountGR, modelProbabilityGR = updateModel(modelCountGR, modelProbabilityGR, w1, w2)

    for w1, w2 in bigrams(line.split()):
        probability += math.log(modelProbabilityGR[w1][w2])

    test_results[i]['GR'] = probability

    print(str(i+1) + " " + max(test_results[i], key=test_results[i].get))


