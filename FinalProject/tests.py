import pandas as pd

#data = pd.read_csv("./kaggle_sentiment140/random_training.csv",delimiter=',')
#print(data.columns)
#print(data.groupby('0').size())

from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = './glove.twitter.27B/glove.twitter.27B.200d.txt'
word2vec_output_file = './glove.twitter.27B/word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
