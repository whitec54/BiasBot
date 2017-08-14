import numpy as np
import json
import string 
import re
from classifiers.bad_words import badWords
from classifiers.model import Model 
import math



class LogisticRegression(Model):
	def __init__(self,feature_name="body",label_name="topic",ngramLen=1):
		self.feature_name = feature_name
		self.label_name = label_name
		self.ngramLen = ngramLen

	def train(self,X,y,alpha=.0009,iters = 500000):
		m,n = X.shape
		self.big_theta = np.zeros((n,len(self.label_key_vector)))

		for index,label in enumerate(self.label_key_vector):
			print("training on "+label)
			#slice out cur col
			cur_little_theta = self.big_theta[:,index]
			cur_y = y[:,index]

			self.big_theta[:,index] = self.descend(cur_little_theta,X,cur_y,alpha,iters)


	def test(self,X,y):
		m,n = X.shape
		preds = self.matrixPredict(X)
		bool_array = np.ones(m, dtype=bool)

		for index in range(m):
			bool_array[index] = np.array_equal(preds[index], y[index])

		correctCount = np.sum(bool_array)

		return correctCount/m

	def learningCurve(self,X_train,y_train,X_cv,y_cv,alpha=.0009,iters = 500000):
		m,n = X_train.shape
		nthIters = math.floor(iters/20);
		dummy_theta = np.zeros((n,1))

		trainCost = self.cost(X_train,y_train,dummy_theta)
		cvCost = self.cost(X_cv,y_cv,dummy_theta)

		print("Train Cost: "+str(trainCost))
		print("CV Cost: "+str(cvCost))
		print()

		for i in range(20):
			dummy_theta = self.descend(dummy_theta,X_train,y_train,alpha,nthIters)
			trainCost = self.cost(X_train,y_train,dummy_theta)
			cvCost = self.cost(X_cv,y_cv,dummy_theta)

			print("Train Cost: "+str(trainCost))
			print("CV Cost: "+str(cvCost))
			print()

	def matrixPredictRaw(self,X):
		theta = self.big_theta
		predsRaw = self.sigmoid(X.dot(theta))

		return predsRaw

	def matrixPredict(self,X):
		predsRaw = self.matrixPredictRaw(X)
		maxs = np.argmax(predsRaw,axis = 1)

		m,n = predsRaw.shape
		preds = np.zeros([m,n])

		for exampleIndex,curMaxInd in enumerate(maxs):
			preds[exampleIndex][curMaxInd] = 1

		return preds

	def textPredictRaw(self,text):
		n = len(self.feature_key_vector)
		textVector = np.zeros([1,n])

		ngrams = super().gen_ngrams(text.split(),self.ngramLen)

		for ngram in ngrams:
			if(ngram in (self.feature_key_vector)):
				ind = self.feature_key_vector.index(ngram)
				textVector[0][ind] += 1

		confidence_mat = self.matrixPredictRaw(textVector)

		return confidence_mat[0][0]

	def textPredict(self,text):
		n = len(self.feature_key_vector)
		textVector = np.zeros([1,n])

		ngrams = super().gen_ngrams(text.split(),self.ngramLen)

		for ngram in ngrams:
			if(ngram in (self.feature_key_vector)):
				ind = self.feature_key_vector.index(ngram)
				textVector[0][ind] += 1

		confidence_mat = self.matrixPredictRaw(textVector)
		max_ind = int(np.argmax(confidence_mat,axis=1))
		return self.label_key_vector[max_ind]

	def sigmoid(self,matrix):
		matrix = matrix * -1
		matrix = np.exp(matrix)
		matrix = np.add(matrix,1)
		matrix = np.power(matrix,-1)
		return matrix

	def next_theta(self,X,y,theta,alpha):
		m = len(y)

		preds = self.sigmoid(X.dot(theta))
		errors = np.subtract(preds,y)
		errors = np.transpose(errors)
		errorSums = errors.dot(X)
		errorSums = np.transpose(errorSums)
		gradient_step = ((alpha/m) * errorSums)
		nextTheta = np.subtract(theta,gradient_step)

		return nextTheta


	def cost(self,X,y,theta):
		m,n = X.shape
		preds = self.sigmoid(X.dot(theta));
		Err = ((-1 *y) * np.log(preds)) - ((1-y) * np.log(1-preds));

		J = np.sum(Err) / (m);
		return J

	def descend(self,init_theta,X,y,alpha,itters):
		temp_theta = init_theta
		for i in range(itters):
			temp_theta = self.next_theta(X,y,temp_theta,alpha)

		return temp_theta

	def save_trained_data(self,filename):
		np.save(filename,self.big_theta)

	def load_trained_data(self,infileName):
		self.big_theta = np.load(infileName)

	def load_feature_key_vector(self,infileName):
		self.feature_key_vector = (np.load(infileName)).tolist()


	def load_label_key_vector(self,infileName):
		self.label_key_vector = list((np.load(infileName)).tolist()) #why are we converting it twice? because we have to because it's bad and I'm lazy

	def getMatrices(self,filename):
		return super().getMatrices(filename,self.ngramLen)



				

def testLogReg():
	classifier = LogisticRegression()
	X_train,X_cv,X_test,y_train,y_cv,y_test = classifier.getMatrices("testdocs.json")

	print(classifier.feature_key_vector)
	print(X_train)
	print()
	print(X_cv)
	print()
	print(X_test)
	print()
	print("***************************************")
	print(classifier.label_key_vector)
	print()
	print(y_train)
	print()
	print(y_cv)
	print()
	print(y_test)
	print()
	print("***************************************")
	print()
	
	

	classifier.train(X_train,y_train)
	accuracy = classifier.test(X_test,y_test)
	print(accuracy)
	print()
	
	dummy_text = "love love love, it's great"

	classifier.learningCurve(X_train,y_train,X_cv,y_cv)

	print(classifier.textPredict(dummy_text))

	# print("dummy doc test:")
	# print("For body: "+dummy_doc["body"])
	# print("Trained on y_train['positive'] bool prediction is:")
	# print(bool_prediction)

	#classifier.learningCurve(X_train,y_train,X_cv,y_cv)
