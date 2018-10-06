from collections import Counter, defaultdict


brown_tagged_train = open("./brown.train.tagged.txt").read().lower()

tokens = set(brown_tagged_train.split())
vocabulary = set([v.split('/')[0] for v in tokens])

wordClasses = defaultdict(lambda: defaultdict(lambda: 0))
for w in brown_tagged_train.split():
    wordClasses[w.split('/')[0]][w.split('/')[1]] += 1

def getMajorityClass(word):
    if (not vocabulary.__contains__(word)):
        return 'nn'

    return max(wordClasses[word], key=wordClasses[word].get)



test = open('brown.test.tagged.txt').read().lower()

match = 0
for i,token in enumerate(test.split()):

    word = token.split('/')[0]
    tag = token.split('/')[1]

    if(getMajorityClass(word) == tag):
        match = match + 1

    #if (i == 1000):
     #   break

print("The accuaracy is :" + str(match/i))
