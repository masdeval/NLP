from collections import Counter, defaultdict


brown_tagged_train = open("./brown.train.tagged.txt").read().lower()

tokens = Counter(brown_tagged_train.split())
vocabularyCounter = Counter([v.split('/')[0] for v in tokens.keys()])
#tagCounter = Counter([v.split('/')[1] for v in tokens.keys()])



wordClasses = defaultdict(lambda: defaultdict(lambda: 0))
for w in brown_tagged_train.split():
    wordClasses[w.split('/')[0]][w.split('/')[1]] += 1

def getMajorityClass(word):
    if (not vocabularyCounter.keys().__contains__(word)):
        return 'nn'

    return max(wordClasses[word], key=wordClasses[word].get)


# def getMajorityClass(word):
#
#     if (not vocabularyCounter.keys().__contains__(word)):
#         return 'nn'
#     tag = ''
#     count = 0
#     for w in [key for key,value in tokens.items() if key.startswith(word+'/')]:
#         if (tokens.get(w) > count):
#             count = tokens.get(w)
#             tag = w.split('/')[1]
#     return tag

test = open('brown.test.tagged.txt').read().lower()

match = 0
for i,token in enumerate(test.split()):

    word = token.split('/')[0]
    tag = token.split('/')[1]

    if(getMajorityClass(word) == tag):
        match = match + 1

    if (i == 1000):
        break

print("The accuaracy is :" + str(match/i))
