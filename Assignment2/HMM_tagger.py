###
# HMM Tagger - Training and decoding using a HMM model
#
# supervisedTraining() - Compute the transition table A and the emission B based on a labeled training set.
#
# decoding() - Estimate the most likely sequence of tags given an input of sequence of words O and a HMM automaton.
#

from collections import Counter, defaultdict
import numpy, math

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
        word = words[i].split('/')[0]
        tag = words[i].split('/')[1]
        if (words[i] == '<F>/<F>'):
            break
        A[tag][words[i+1].split('/')[1]]  += 1
        B[tag][word] += 1
        vocabulary.add(words[i].split('/')[0])

        if(word == '.'):
            startProbability[words[i+1].split('/')[1]] += 1

    return A,B,vocabulary,startProbability

def transitionProbability(q_from,q_to,A):
    if (A[q_from][q_to] == 0):
        return 0
    else:
        return math.log(A[q_from][q_to]/sum(A[q_from].values()))

def emissionProbability(o, q, B, vocabulary, numberStates):
    if(B[q].__contains__(o)):
        return math.log(B[q][o]+1/(sum(B[q].values())+len(vocabulary)))
    elif (vocabulary.__contains__(o)):
        return math.log(1/(sum(B[q].values())+len(vocabulary)))
    else:
        return math.log(1/numberStates)


def startProbability(q,start):
    if (not start.__contains__(q)):
        return 0
    else:
        return math.log(start[q]/sum(start.values()))

# Input: list of observations obs, transition probabilities A, emission probabilities B
# Output: most probable sequence of states
def viterbi(obs, A, B, vocabulary, start):

    viterbiMatrix = numpy.zeros((len(A),len(obs)))
    states = list(A.keys())

    #Initialization
    for i in range(len(A)-1):
        viterbiMatrix[i][0] = startProbability(states[i],start) + emissionProbability(obs[0], states[i], B, vocabulary, len(states))

    for i in range(1, len(obs)-1):
        for j in range (len(A)-1):
            viterbiMatrix[j][i] = max([v+transitionProbability(states[s],states[j],A) for s,v in enumerate(viterbiMatrix[:,i-1])]) + emissionProbability(obs[i],states[j],B, vocabulary,len(states))

    bestSequence = [states[numpy.argmax(viterbiMatrix[:,i])] for i in range(len(obs)) ]

    return bestSequence


A,B,vocabulary,start = supervisedTraining('./brown.train.tagged.txt')

test = open('./brown.test.tagged.txt').read().lower()

match = 0
words = list()
tags = list()
for i,token in enumerate(test.split()):
    words.append(token.split('/')[0])
    tags.append(token.split('/')[1])
    #if (i == 1000):
     #   break


result = viterbi(words,A,B,vocabulary,start)

for i,tag in enumerate(tags):
    if (tag == result[i]):
        match += 1


print("The accuaracy is :" + str(match/i))
