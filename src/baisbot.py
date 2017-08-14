from classifiers.logistic import LogisticRegression
import utils.article_getter5000 as ag

class BiasBot:
	def __init__(self):
		self.sentiment_theta_path = "thetas/LogRegSentimentTheta.npy"
		self.sentiment_feature_key_path = "thetas/LogRegSentimentFeatureKey.npy"
		self.sentiment_label_key_path = "thetas/LogRegSentimentLabelKey.npy"

		self.label_theta_path = "thetas/LogRegWorldPoliticsTheta.npy"
		self.label_feature_key_path = "thetas/LogRegWorldPoliticsFeatureKey.npy"
		self.label_label_key_path = "thetas/LogRegWorldPoliticsLabelKey.npy"
	
	#returns [{site:fox,sentiment:.85},{site:cnn,sentiment:.99},...ect ]
	def getDisplayData(self,url):
		headline = self.getHeadline(url)
		label = self.getLabel(headline)
		articles = self.getRelatedArticles(label)

		displayData = []
		for article in articles:
			data = {"site":"","sentiment":0}

			data["site"] = article["site"]
			data["sentiment"] = self.getSentiment(article["sentences"])

			displayData.append(data)

		return displayData

	def getHeadline(self,url):

		#TODO
		#need to actually write this damn part

		return "syria syria syria syria syria syria syria"

	def getLabel(self,headline):

		classifier = LogisticRegression()

		classifier.load_trained_data(self.label_theta_path)
		classifier.load_feature_key_vector(self.label_feature_key_path)
		classifier.load_label_key_vector(self.label_label_key_path)

		label = classifier.textPredict(headline)

		print("label " + label)

		return label

	def getRelatedArticles(self,label):
		#need to manually make the labels better for search so for now...
		label = "syria"
		
		related_articles = ag.get_articles(label)
		return related_articles

	def getSentiment(self,sentences):

		classifier = LogisticRegression()

		classifier.load_trained_data(self.sentiment_theta_path)
		classifier.load_feature_key_vector(self.sentiment_feature_key_path)
		classifier.load_label_key_vector(self.sentiment_label_key_path)
		
		sentiment = 0
		for sentence in sentences:
			sentiment += classifier.textPredictRaw(sentence)
			return sentiment

		sentiment = sentiment/len(sentences)
		
		return sentiment

	
		