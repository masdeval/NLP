from collections import Counter, defaultdict


brown_tagged_train = open("./brown.train.tagged").read().lower()

tokens = Counter(brown_tagged_train.split())
vocabularyCounter = Counter([v.split('/')[0] for v in tokens.keys()])
#tagsCounter = Counter([v.split('/')[1] for v in tokens.keys()])

# wordClasses = defaultdict(lambda: defaultdict(lambda: 0))
# for w in vocabularyCounter.keys():
#     for tag in [key for key,value in tokens.items() if key.startswith(w+'/')]:
#         wordClasses[w][tag.split('/')[1]] = tokens.get(tag)
#
#
# def getMajorityClass(word):
#
#     if (not vocabularyCounter.keys().__contains__(word)):
#         return 'nn'
#
#     return max(wordClasses[word], key=wordClasses[word].get())

wordClasses = defaultdict(lambda: defaultdict(lambda: 0))
for w in brown_tagged_train.split():
    wordClasses[w.split('/')[0]][w.split('/')[1]] += 1

def getMajorityClass2(word):
    if (not vocabularyCounter.keys().__contains__(word)):
        return 'nn'

    return max(wordClasses[word], key=wordClasses[word].get)


def getMajorityClass(word):

    if (not vocabularyCounter.keys().__contains__(word)):
        return 'nn'
    tag = ''
    count = 0
    for w in [key for key,value in tokens.items() if key.startswith(word+'/')]:
        if (tokens.get(w) > count):
            count = tokens.get(w)
            tag = w.split('/')[1]
    return tag

print(getMajorityClass('again'))

print([key for key,value in tokens.items() if key.startswith('again/')])
print(tokens.get('again/rb'))
print(tokens.get('again/rb-hl'))
print(tokens.get('again/rb-tl'))

test = open('brown.test.tagged').read().lower()

match = 0
for i,token in enumerate(test.split()):

    word = token.split('/')[0]
    tag = token.split('/')[1]

    if(getMajorityClass2(word) == tag):
        match = match + 1

print(match)
print(i)