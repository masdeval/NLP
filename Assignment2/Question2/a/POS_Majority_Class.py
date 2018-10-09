from collections import Counter, defaultdict


brown_tagged_train = open("./brown.train.tagged.txt").read().lower()

tokens = set(brown_tagged_train.split())
vocabulary = set([v.rsplit('/',1)[0] for v in tokens])

wordClasses = defaultdict(lambda: defaultdict(lambda: 0))
for w in brown_tagged_train.split():
    wordClasses[w.rsplit('/',1)[0]][w.rsplit('/',1)[1]] += 1

def getMajorityClass(word):
    if (not vocabulary.__contains__(word)):
        return 'nn'

    return max(wordClasses[word], key=wordClasses[word].get)



test = open('brown.test.tagged.txt').read().lower()

match = 0
for i,token in enumerate(test.split()):

    word = token.rsplit('/',1)[0]
    tag = token.rsplit('/',1)[1]

    if(getMajorityClass(word) == tag):
        match = match + 1

print("The accuaracy is :" + str(match/i))
