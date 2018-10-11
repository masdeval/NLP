###
# HMM Tagger - Training and decoding using a HMM model
#
# supervisedTraining() - Compute the transition table A and the emission B based on a labeled training set.
#
# decoding() - Estimate the most likely sequence of tags given an input of sequence of words O and a HMM automaton.
#

from collections import defaultdict
import numpy, math, time

def supervisedTraining(file):

    brown_tagged_train = open(file).read().lower()

    A = defaultdict(lambda: defaultdict(lambda: 0))
    B = defaultdict(lambda: defaultdict(lambda: 0))
    startProbability = defaultdict(lambda: 0)

    words = brown_tagged_train.split()
    vocabulary = set()
    words.append('<F>/<F>')

    # Calculate A, B
    for i in range(len(words)):

        if(not words[i].__contains__('/')):
            continue

        word = words[i].rsplit('/',1)[0]
        tag = words[i].rsplit('/',1)[1]
        vocabulary.add(word)
        if (words[i] == '<F>/<F>'):
            break
        A[tag][words[i+1].rsplit('/',1)[1]]  += 1
        B[tag][word] += 1

        if(tag == '.' ):
            startProbability[words[i+1].rsplit('/',1)[1]] += 1

    return A,B,vocabulary,startProbability

def transitionProbability(q_from,q_to,A):
    if (A[q_from][q_to] == 0):
        return 0
    else:
        return math.log(A[q_from][q_to]/sum(A[q_from].values()))

def emissionProbability(o, q, B, vocabulary):
    if(B[q][o] != 0):
        return math.log(B[q][o]+1/(sum(B[q].values())+len(vocabulary))) #AddOne Smoothing
        #return math.log(B[q][o]/sum(B[q].values()))
    elif (vocabulary.__contains__(o)):
        return math.log(1/(sum(B[q].values())+len(vocabulary)))  #AddOne Smoothing
        #return -numpy.inf # The word is in the vocabulary and will be selected in some other tag. -inf because using log and summing
    else:
        return math.log(1/len(vocabulary)) # the word is not in the vocabulary and we can't leave it with zero probability


def startProbability(q,start):
    if (not start.__contains__(q)):
        return -numpy.inf
    else:
        return math.log(start[q]/sum(start.values()))

# Implementation of the Viterbi algorithm with Beam Search
# Input: list of observations obs, transition probabilities A, emission probabilities B
# Output: most probable sequence of states

def handleMax(i,value,topElements,size):

    if len(topElements) < size:
        topElements[i] = value
        return topElements

    for v in topElements.keys():
        if (topElements[v] < value):
            topElements.pop(v)
            topElements[i] = value
            assert(len(topElements) == size)
            break
    return topElements

def decoding(obs, A, B, vocabulary, start):

    viterbiMatrix = numpy.zeros((len(A),len(obs)))
    states = list(A.keys())
    top = dict()

    #Initialization
    for i in range(len(A)):
        viterbiMatrix[i][0] = startProbability(states[i],start) + emissionProbability(obs[0], states[i], B, vocabulary)
        top = handleMax(i,viterbiMatrix[i][0],top,5)

    for i in range(1, len(obs)):
        statesToVisit = top.keys()
        top = dict()
        for j in range (len(A)):
            max_ = list()
            for s in statesToVisit:
                max_.append(viterbiMatrix[s, i - 1] + transitionProbability(states[s], states[j], A))

            viterbiMatrix[j][i] =  max(max_)+ emissionProbability(obs[i],states[j],B, vocabulary)
            top = handleMax(j,viterbiMatrix[j][i],top,5)

    bestSequence = [states[numpy.argmax(viterbiMatrix[:,i])] for i in range(len(obs)) ]

    return bestSequence


A,B,vocabulary,start = supervisedTraining('./brown.train.tagged.txt')

# Evaluation
begin = time.perf_counter()
test = open('./brown.test.tagged.txt')
match = 0
numberOfWords = 0
for sentence in (test):
    words = list()
    tags = list()

    for i, token in enumerate(sentence.lower().split()):
        words.append(token.rsplit('/', 1)[0])
        tags.append(token.rsplit('/', 1)[1])
        numberOfWords += 1

    result = decoding(words, A, B, vocabulary, start)

    for i, tag in enumerate(tags):
        if (tag == result[i]):
            match += 1

    #if (numberOfWords > 3000):
     #   break

end = time.perf_counter()
print("The accuaracy is :" + str(match / numberOfWords))
print("\n Time elapsed: " + str(end - begin))