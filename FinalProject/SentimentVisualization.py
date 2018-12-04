import re
import tweepy
from tweepy import OAuthHandler
from joblib import dump, load
from gensim.models import KeyedVectors
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

import numpy as np
import preprocess_twitter as stanfordPreprocessing

class TwitterClient(object):
	'''
	Generic Twitter Class for sentiment analysis.
	'''
	def __init__(self):
		'''
		Class constructor or initialization method.
		'''
		# keys and tokens from the Twitter Dev Console
		consumer_key = 'xFBZEWKwyqUzAXhNdYcPRVKmu'
		consumer_secret = 'QPUvRZtxKDLeTc1DjN5MW4AeCMzKOeEsZN3rCZoW5re3pJoSIv'
		access_token = '44031454-Xl9oKsUcnZY7vuQMORfqdhY8eh2MjVx9Gyrrgo8aq'
		access_token_secret = 'Sp2TiSnNdjZT7CUoKBIdOOMtYrO5H3MOCOM03nZt1L97e'

		# attempt authentication
		try:
			# create OAuthHandler object
			self.auth = OAuthHandler(consumer_key, consumer_secret)
			# set access token and secret
			self.auth.set_access_token(access_token, access_token_secret)
			# create tweepy API object to fetch tweets
			self.api = tweepy.API(self.auth)
		except:
			print("Error: Authentication Failed")

	def clean_tweet(self, tweet):
		'''
		Utility function to clean tweet text by removing links, special characters
		using simple regex statements.
		'''
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

	def buildTwitterVector(self, tokens, word2vec, vectorSize):
		vec = np.zeros(vectorSize)
		count = 0.
		for word in tokens:
			try:
				vec += word2vec[word]
				count += 1.
			except KeyError:  # handling the case where the token is not present
				continue
		if count != 0:
			vec /= count

		assert (len(vec) == vectorSize)
		return vec

	def get_tweet_sentiment(self, tweet, word2vec):
		'''
		Utility function to classify sentiment of passed tweet
		using textblob's sentiment method
		'''

		clf = load('svm_model.save')
		feature = self.buildTwitterVector(tweet,word2vec,vectorSize=200)
		analysis = clf.predict(feature.reshape(1, -1))

		# set sentiment
		if analysis > 0:
			return 'positive'
		else:
			return 'negative'

	def get_tweets(self, query, count = 10):
		'''
		Main function to fetch tweets and parse them.
		'''
		# empty list to store parsed tweets
		tweets = []

		try:
			# call twitter api to fetch tweets
			fetched_tweets = self.api.search(q = query, count = count)
			# Load Glove word2vec
			#word2vec = KeyedVectors.load_word2vec_format("./glove.twitter.27B/word2vec200d.txt")
			word2vec = KeyedVectors.load("./twitter_vectors.model", mmap='r')

			# parsing tweets one by one
			for tweet in fetched_tweets:
				# empty dictionary to store required params of a tweet
				parsed_tweet = {}

				# saving text of tweet
				parsed_tweet['text'] = tweet.text
				#tweet_ = stanfordPreprocessing.tokenize(tweet.text).split()
				tweet_ = gensim.utils.simple_preprocess(tweet.text)
				# saving sentiment of tweet
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet_, word2vec)

				# appending parsed tweet to tweets list
				if tweet.retweet_count > 0:
					# if tweet has retweets, ensure that it is appended only once
					if parsed_tweet not in tweets:
						tweets.append(parsed_tweet)
				else:
					tweets.append(parsed_tweet)

			# return parsed tweets
			return tweets

		except tweepy.TweepError as e:
			# print error (if any)
			print("Error : " + str(e))

def main():
	# creating object of TwitterClient Class
	api = TwitterClient()
	# calling function to get tweets
	tweets = api.get_tweets(query = 'christmas', count = 100)
	# picking positive tweets from tweets
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
	# percentage of positive tweets
	print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
	# picking negative tweets from tweets
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
	# percentage of negative tweets
	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
	# percentage of neutral tweets
	#print("Neutral tweets percentage: {} % ".format(100*len(tweets - ntweets - ptweets)/len(tweets)))

	# printing first 5 positive tweets
	print("\n\nPositive tweets:")
	for tweet in ptweets[:10]:
		print(tweet['text'])

	# printing first 5 negative tweets
	print("\n\nNegative tweets:")
	for tweet in ntweets[:10]:
		print(tweet['text'])

if __name__ == "__main__":
	# calling main function
	main()
