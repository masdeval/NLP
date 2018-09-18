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

# Implement a letter bigram model, which learns letter bigram
# probabilities from the training data. A separate bigram model has to be learned for
# each language (from each of the training files provided). Apply the models to
# determine the most likely language for each sentence in the test file (that is,
# determine the probability associated with each sentence in the test file, using each
# of the four language models).
#

from nltk import bigrams
from collections import defaultdict
import re

models = dict()
models['EN'] = dict()
models['FR'] = dict()
models['GR'] = dict()

def makeLetterModel(fileName):

 sentences = open(fileName).read()
 #lowering case
 sentences = sentences.lower()
 #removing punctuation
 sentences = re.sub(r"\W", ' ', sentences)
 #replace sequences of blank characteres to a single one
 sentences = re.sub(r"\s+", ' ', sentences)

 #vocabulary_EN = Counter(re.findall(r'\w+', sentences_EN))

 bigram = bigrams(sentences)

 # model[w1][w2] stores the number of times each bigram w1,w2 was seen
 model = defaultdict(lambda : defaultdict(lambda:0))

 # counting bigrams
 for w1, w2 in bigram:
        model[w1][w2] += 1

 # getting probabilities
 for w1 in model:
    total_count = (sum(model[w1].values())) #number of times a bigram begining with w1 has ocured
    for w2 in model[w1]:
        model[w1][w2] /= total_count # calculating the relative frequency for each bigram

 return model

models['EN'] = makeLetterModel('./EN.txt')
models['FR'] = makeLetterModel('./FR.txt')
models['GR'] = makeLetterModel('./GR.txt')

import math
test_file = open('./LangID.test.txt')

test_results = defaultdict(lambda : defaultdict( lambda : float))

for i,line in enumerate(test_file):
    line = line.lower()
    line = re.sub(r"\W", ' ', line)
    line = re.sub(r"\s+", ' ', line)
    line = re.sub(r'^\d+\s','',line) # remove the initial number from each row

    #EN
    probability = 0.0
    for w1,w2 in bigrams((line)):
        if(models['EN'][w1][w2] == 0):
            #print(model_EN[w1].items())
            continue
        probability += math.log(models['EN'][w1][w2])

    test_results[i]['EN'] = probability

    # FR
    probability = 0.0
    for w1, w2 in bigrams(line):
        if (models['FR'][w1][w2] == 0):
            continue
        probability += math.log(models['FR'][w1][w2])

    test_results[i]['FR'] = probability

    # GR
    probability = 0.0
    for w1, w2 in bigrams(line):
        if (models['GR'][w1][w2] == 0):
            # print(model_EN[w1].items())
            continue
        probability += math.log(models['GR'][w1][w2])

    test_results[i]['GR'] = probability


for item in test_results:
    print(str(item+1) + " " + max(test_results[item], key=test_results[item].get))

