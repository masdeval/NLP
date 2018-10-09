from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import re

def phaseOneTest():
    # Train
    brown_tagged_train = open("./brown.train.tagged.txt").read().lower()
    tokens = Counter(brown_tagged_train.split())
    vocabularyCounter = Counter([v.rsplit('/', 1)[0] for v in tokens.keys()])
    wordClasses = defaultdict(lambda: defaultdict(lambda: 0))
    for w in brown_tagged_train.split():
        wordClasses[w.rsplit('/', 1)[0]][w.rsplit('/', 1)[1]] += 1

    def getMajorityClass(word):
        if (not vocabularyCounter.keys().__contains__(word)):
            return 'nn'
        return max(wordClasses[word], key=wordClasses[word].get)

    test = open('brown.test.tagged.txt').read().lower()

    hits_miss = defaultdict(lambda: defaultdict(lambda: 0))

    # keep track of the misses and hits to build a confusion matrix
    for i, token in enumerate(test.split()):
        word = token.rsplit('/', 1)[0]
        tag = token.rsplit('/', 1)[1]
        # keep track of the misses and hits to build a confusion matrix
        hits_miss[tag][getMajorityClass(word)] += 1

    # sort by relevance
    for w in hits_miss:
        hits_miss[w] = sorted(hits_miss[w].items(), key=lambda kv: kv[1], reverse=True)

    for tag1 in hits_miss:
        # print only the cases where were mismatch classification
        if (len(hits_miss[tag1]) > 1):
            count = Counter(hits_miss[tag1]).most_common(5)
            print("Tag: " + tag1 + " taggings: " + str(count))




#### Starting the tests to identify possible enhacement strategies

# Shows the hits and misses of the majority tagging strategy
phaseOneTest()

# Based on the results, I chose the following five tags to verify:

# in (prepostion) tagged as to (infinitive marker) -> change TO to IN if next tag is different of VB

# nns tagged as nn -> while the regular plural is spelled -s after most nouns, it is spelled -es after
# words ending in -s (ibis/ibises), -z (waltz/waltzes), -sh (thrush/thrushes), -ch (finch/finches),
# and sometimes -x (box/boxes). Nouns ending in -y preceded by a consonant change the -y to -i (butterfly/butterflies

# vb tagged as nn -> change NN to VB where previous tag is TO

# nn tagged as vb -> change VB to NN where previous tag is not TO

# vbg tagged as nn -> change NN to VBG where suffix is .ING

# Train
brown_tagged_train = open("./brown.train.tagged.txt").read().lower()
tokens = Counter(brown_tagged_train.split())
vocabularyCounter = Counter([v.rsplit('/',1)[0] for v in tokens.keys()])
wordClasses = defaultdict(lambda: defaultdict(lambda: 0))
for w in brown_tagged_train.split():
    wordClasses[w.rsplit('/',1)[0]][w.rsplit('/',1)[1]] += 1

def getMajorityClass(word):
    if (not vocabularyCounter.keys().__contains__(word)):
        return 'nn'
    return max(wordClasses[word], key=wordClasses[word].get)

test = open('brown.test.tagged.txt').read().lower()
match = 0
words = test.split()

for i,token in enumerate(words):

    word = token.rsplit('/',1)[0]
    tag = token.rsplit('/',1)[1]

    #Rule 1: change TO to IN if next tag is different of VB
    if (getMajorityClass(word) == 'to'):
        if (getMajorityClass(words[i+1].rsplit('/',1)[0]) != 'vb'):
            if ('in' == tag):
                match = match + 1
                continue

    #Rule 2: nns tagged as nn
    if (getMajorityClass(word) == 'nn'):
        if (re.match(r'\w+ses$',word) or re.match(r'\w+zes$',word) or re.match(r'\w+shes$',word) or re.match(r'\w+ches$',word) or re.match(r'\w+ies$',word) or re.match(r'\w+s$',word)):
            if ('nns' == tag):
                match = match + 1
                continue

    #Rule 3: change NN to VB where previous tag is TO
    if (getMajorityClass(word) == 'nn'):
        if (words[i - 1].rsplit('/', 1)[1] == 'to'):
            if ('vb' == tag):
                match = match + 1
                continue

    #Rule 4: change VB to NN where previous tag is not TO
    if (getMajorityClass(word) == 'vb'):
        if (words[i - 1].rsplit('/', 1)[1] != 'to'):
            if ('nn' == tag):
                match = match + 1
                continue

    #Rule 5:change NN to VBG where suffix is .ING
    if (getMajorityClass(word) == 'nn'):
        if (re.match(r'\w+ing$', word) ):
            if ('vbg' == tag):
                match = match + 1
                continue

    if(getMajorityClass(word) == tag):
        match = match + 1


print("The accuaracy is :" + str(match/i))