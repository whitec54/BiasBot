import json
import string
from classifiers.bad_words import badWords
import re
import numpy as np
import math

class Model:

	def __init__(self):
		self.trainedParameters = []

	#overload 'for doc in json.load(file): modifty trained params'
	def train(self,file,X,y):
		self.trainedParameters = "Define self.trained parameters with subModel -Model"

	#overload 'for doc in json.load(file): call predict and check'
	def test(self,file,X,y):
		successRate = 0
		print("success rate: " + str(successRate))

	#call on trained params only, very little computation
	def predict(self,document,X,y):
		return "label = overload this"


	def save_trained_data(self,outfileName):
		with open(outfileName, 'w') as outfile:
			json.dump(self.trainedParameters, outfile)

	def load_trained_data(self,infileName):
		with open(infileName,'r') as infile:
			self.trainedParameters = json.load(infile)


	def clean(self,text):
		toRemove = badWords()

		cleaned = ' '.join(word.strip(string.punctuation).lower() for word in text.split())#puntcuation, capitals
		cleaned = re.sub('<.*?>', ' ', cleaned) #html

		scripts = re.compile(r'<script[\s\S]+?/script>')
		cleaned = re.sub(scripts, "", cleaned)

		style = re.compile(r'<style[\s\S]+?/style>')
		cleaned = re.sub(style, "", cleaned)

		cleaned = cleaned.split()
		cleaned = [word for word in cleaned if word not in toRemove.words] # useless words

		return cleaned


	def gen_ngrams(self,words,ngramLen):
		ngrams=[]

		for i in range(len(words)):
			ngram = ""
			if(ngramLen>1 and (i-(ngramLen-1)) >=0 ):
				for j in range(i-(ngramLen-1),i+1):
					ngram += words[j]
			elif(ngramLen == 1):
				ngram = words[i]

			if(ngram != ""):
				ngrams.append(ngram)

		return ngrams


	def gen_ngram_to_count(self,data,feature_name,ngramLen):
		ngram_to_count = {}
		for doc in data:
			words = self.clean(doc[feature_name])
			ngrams = self.gen_ngrams(words,ngramLen)
			
			for ngram in ngrams:
				if ngram in ngram_to_count:
					ngram_to_count[ngram] += 1
				elif(ngram):
					ngram_to_count[ngram] = 1

		return ngram_to_count


	def get_unique_labels(self, data):
		labels = set()
		label_name = self.label_name

		for doc in data:
			labels.add(doc[label_name])

		return labels


	def gen_ngram_key_vector(self, data,ngramLen=1,remove_count = 120000):
		ngram_count_list = []
		key_vector = []

		feature_name = self.feature_name
		label_name = self.label_name

		
		ngram_to_count = self.gen_ngram_to_count(data,feature_name,ngramLen)
		labels = self.get_unique_labels(data)

		#convert to array of tuples
		for key, value in ngram_to_count.items():
			temp = (key,value)
			ngram_count_list.append(temp)

		#sort it by count
		ngram_count_list = sorted(ngram_count_list, key=lambda x: x[1],reverse=True)

		#get just the words
		[key_vector.append(pair[0]) for pair in ngram_count_list]

		#chop down to size, keeping old len for printing
		og_len = len(key_vector)
		cap = len(key_vector)-remove_count
		key_vector = key_vector[0:cap]

		print("generated "+str(og_len)+" "+str(ngramLen)+"-grams")
		print("And kept the most common " + str(og_len-remove_count))

		return key_vector,labels

	def genWholeMatrices(self,filename,ngramLen=1):
		skip_num = 2
		with open(filename,'r') as infile:
			data = json.load(infile)

		key_vector,labels = self.gen_ngram_key_vector(data,ngramLen)
		self.label_key_vector = list(labels)
		self.feature_key_vector = key_vector

		feature_name = self.feature_name
		label_name = self.label_name

		m = math.floor(len(data)/skip_num) + skip_num
		n = len(key_vector)
		X = np.zeros([m,n])

		#init y examples by num labels 
		y = np.zeros([m,len(self.label_key_vector)])
		cur_index = 0;
		for k,doc in enumerate(data):
			if k%skip_num != 0:
				continue


			row = np.zeros(n)
			words = self.clean(doc[feature_name])
			ngrams = self.gen_ngrams(words,ngramLen)

			#build X row
			for ngram in ngrams:
				if(ngram in (key_vector)):
					ind = key_vector.index(ngram)
					row[ind] += 1

			X[cur_index] = row

			#modify y row
			cur_label = doc[label_name]
			label_index = self.label_key_vector.index(cur_label)
			y[cur_index][label_index] = 1
			cur_index += 1

		
		return X,y


	def shuffleUnison(self,X,y):
		assert(len(X) == len(y))
		permutation = np.random.permutation(len(X))

		X = X[permutation]

		y = y[permutation]

		return X,y

	def splitX(self,X_whole,m):
		trainEndInd = math.floor(m*0.6)
		cvEndInd = math.floor(m*0.8)

		X_train = X_whole[:trainEndInd,:]
		X_cv = X_whole[trainEndInd:cvEndInd,:]
		X_test = X_whole[cvEndInd:,:]

		return X_train,X_cv,X_test

	def splitYDict(self,y_whole,m):
		trainEndInd = math.floor(m*0.6)
		cvEndInd = math.floor(m*0.8)

		y_train = y_whole[:trainEndInd,:]
		y_cv = y_whole[trainEndInd:cvEndInd,:]
		y_test = y_whole[cvEndInd:,:]

		return y_train,y_cv,y_test

	def getMatrices(self,filename,ngramLen=1):
		X_whole,y_whole = self.genWholeMatrices(filename,ngramLen)
		
		X_whole,y_whole = self.shuffleUnison(X_whole,y_whole)

		m,n = X_whole.shape
		X_train,X_cv,X_test = self.splitX(X_whole,m)

		y_train,y_cv,y_test = self.splitYDict(y_whole,m)

		return X_train,X_cv,X_test,y_train,y_cv,y_test