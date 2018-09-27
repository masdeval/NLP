###
# HMM Tagger - Training and decoding using a HMM model
#
# supervisedTraining() - Compute the transition table A and the emission B based on a labeled training set.
#
# decoding() - Estimate the most likely sequence of tags given an input of sequence of words O and a HMM automaton.
#

from collections import Counter, defaultdict
import numpy

def supervisedTraining(file):

    brown_tagged_train = open(file).read().lower()

    A = defaultdict(lambda: defaultdict(lambda: 0))
    B = defaultdict(lambda: defaultdict(lambda: 0))

    words = brown_tagged_train.split()
    words.append('<F>/<F>')

    # Calculate A, B
    for i,w in enumerate(words):
        if (w == '<F>/<F>'):
            break
        A[w.split('/')[1]][w[i+1].split('/')[1]]  += 1
        B[w.split('/')[1]][w.split('/')[0]] += 1

    return A,B

def transitionProbability(q_from,q_to,A):
    return A[q_from][q_to]/sum(A[q_from])

def emissionProbability(o, q, B, vocabulary, numberStates):
    if(B[q].__contains__(o)):
        return B[q][o]+1/sum(B[q])+len(vocabulary)
    elif (vocabulary.contains(o)):
        return 1/sum(B[q])+len(vocabulary)
    else:
        return 1/numberStates


def startProbability(q,A):
    return A['.'][q]/sum(A['.'])

# Input: list of observations obs, transition probabilities A, emission probabilities B
# Output: most probable sequence of states
def viterbi(obs, A, B, vocabulary):
    obs = obs.split()
    viterbiMatrix = numpy.zeros((len(A),len(obs)))
    states = list(A.keys());

    #Initialization
    for i in range(len(A)-1):
        viterbiMatrix[i][0] = startProbability(states[i],A) * emissionProbability(obs[0], states[i], B, vocabulary, len(states))

    for i in range(1, len(obs)-1):
        for j in range (len(A)-1):
            for k in range (len(A)-1):


def Viterbit(obs, states, s_pro, t_pro, e_pro):
	path = { s:[] for s in states} # init path: path[s] represents the path ends with s
	curr_pro = {}
	for s in states:
		curr_pro[s] = s_pro[s]*e_pro[s][obs[0]]
	for i in xrange(1, len(obs)):
		last_pro = curr_pro
		curr_pro = {}
		for curr_state in states:
			max_pro, last_sta = max(((last_pro[last_state]*t_pro[last_state][curr_state]*e_pro[curr_state][obs[i]], last_state)
				                       for last_state in states))
			curr_pro[curr_state] = max_pro
			path[curr_state].append(last_sta)

	# find the final largest probability
	max_pro = -1
	max_path = None
	for s in states:
		path[s].append(s)
		if curr_pro[s] > max_pro:
			max_path = path[s]
			max_pro = curr_pro[s]
		# print '%s: %s'%(curr_pro[s], path[s]) # different path and their probability
	return max_path


test = open('brown.test.tagged').read().lower()

match = 0
for i,token in enumerate(test.split()):

    word = token.split('/')[0]
    tag = token.split('/')[1]

    if(getMajorityClass(word) == tag):
        match = match + 1

print("The accuaracy is :" + str(match/i))
