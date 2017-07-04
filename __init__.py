from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords as sw
import string
import random
import os
import warnings

class Classifier:
	""" The main class responsible for classifying sentences into documents.

		The class is responsible for selection of training and test data. It
		uses three different classifiers for training and prediction which are:
		1. NaiveBayes
		2. K Nearest Neighbours
		3. SVM (Support Vector Machines)
	"""
	def __init__(self, docs):
		""" The constructor function responsible for various initializations.

			docs:       list of training documents
			stopwords:  standard list of stopwords of NLTK
			vectorizer: Count Vectorizer which is responsible for bag of words
						representation - applies n-gram technique and removal of
						stopwords along with selection of maximum no. of features.
		"""
		self.docs = docs
		self.stopwords = [str(word) for word in sw.words("english")]
		self.vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=self.stopwords, max_features=2000)
		self.train_data, self.train_labels, self.test_data, self.test_labels = [], [], [], []
		self.tfIdf = None

	def setTestAndTrainingData(self):
		""" Splits the documents into training and test data.

			The splitting is done by a random shuffle in 80:20 ratio for training
			and test data respectively. Also, it sets the training and test labels
			appropriately to be used later while training and prediction.
		"""
		remData = string.punctuation.replace(".", "")
		printable = string.printable
		# pattern = r"[{}]".format(remData)
		replace_punctuation = string.maketrans(remData, ' ' * len(remData))
		for fileNum in range(len(self.docs)):
			fileName = self.docs[fileNum]
			fileContent = open("Data/" + fileName).read().lower()
			data = fileContent.translate(replace_punctuation).lower()
			filter(lambda x: x in printable, data)
			data = list(data)

			i = 0

			while i < len(data):
				try:
					if data[i] == ".":
						if data[i - 1].isdigit() and data[i + 1].isdigit():
							data[i] = " "
					elif data[i] == "[":
						if data[i + 2] == "]":
							data[i: i + 3] = ""
						elif data[i + 3] == "]":
							data[i: i + 4] = ""
					elif data[i] == " ":
						n = i + 1
						while data[n] == " ":
							n = n + 1
						data[i + 1:n] = ""
					elif ord(data[i]) < 32 or ord(data[i]) > 126:
						data[i] = ""
				except Exception:
					pass
				i += 1
				# print i, len(data)

			data = "".join(data).split(".")
			for i in range(len(data)):
				data[i] = data[i].strip()
			data = data[:-1]
			tempData = data
			random.shuffle(tempData)
			length = len(data)
			numTrain = int(length * 0.8)
			trainTemp = tempData[:numTrain + 1]
			testTemp = tempData[numTrain + 1:]
			self.train_data = self.train_data + trainTemp
			self.test_data = self.test_data + testTemp
			self.train_labels = self.train_labels + [fileNum] * len(trainTemp)
			self.test_labels = self.test_labels + [fileNum] * len(testTemp)
		# print len(self.train_data), len(self.test_data)

	def transformFeatures(self):
		""" Transforms the features by applying TfIdf metric.

			Tf  : Term frequency
			Idf : Inverse Document Frequency
		"""
		# print self.train_data
		self.XTrain = self.vectorizer.fit_transform(self.train_data)
		self.tfidfTransformer = TfidfTransformer(use_idf=True).fit(self.XTrain)
		self.XTrainTfIdf = self.tfidfTransformer.transform(self.XTrain)

	def naiveBayesClassifier(self):
		""" Applies the naiveBayesClassifier and stores the predicted result
			in self.predicted.
		"""
		self.clf = MultinomialNB().fit(self.XTrainTfIdf, self.train_labels)
		self.vectorizer.set_params(max_features=2000)
		XTest = self.vectorizer.transform(self.test_data)
		self.XTestTfIdf = self.tfidfTransformer.transform(XTest)
		self.predicted = self.clf.predict(self.XTestTfIdf)

	def KNNClassifier(self):
		""" Applies the K-nearest Neighbours Classifier and stores the predicted
			result in self.predicted.
		"""
		# dt = DistanceMetric(metric="pyfunc", func=self.distance)
		self.clf = KNeighborsClassifier(n_neighbors=341).fit(self.XTrainTfIdf, self.train_labels)
		self.vectorizer.set_params(max_features=2400)
		# print(self.vectorizer.get_params())
		XTest = self.vectorizer.transform(self.test_data)
		self.XTestTfIdf = self.tfidfTransformer.transform(XTest)
		self.predicted = self.clf.predict(self.XTestTfIdf)

	def SVMClassifier(self):
		""" Applies the Support Vector Machine (SVM) Classifier and stores the
			predicted result in self.predicted.
		"""
		self.clf = LinearSVC().fit(self.XTrainTfIdf, self.train_labels)
		self.vectorizer.set_params(max_features=2000)
		XTest = self.vectorizer.transform(self.test_data)
		self.XTestTfIdf = self.tfidfTransformer.transform(XTest)
		self.predicted = self.clf.predict(self.XTestTfIdf)

	def score(self):
		""" Based on the actual test data labels and predicted labels, generates
			the accuracy of the classification and also generates a report for
			the same.
		"""
		print("ACCURACY: " + str(accuracy_score(self.test_labels, self.predicted)))
		print(metrics.classification_report(self.test_labels, self.predicted))

	def testModel(self):
		""" Tests the training model against training and test data and calls
			the score method for full report.
		"""
		self.classifier = raw_input("Enter type of classifier (NB or KNN or SVM): ")
		if self.classifier == "NB":
			self.naiveBayesClassifier()
		elif self.classifier == "KNN":
			self.KNNClassifier()
		elif self.classifier == "SVM":
			self.SVMClassifier()
		self.score()

	def inputPrediction(self):
		""" Takes in the input sentence to be classified and type of classifier
			to be used and classifies it to some appropriate document.
		"""
		self.inp = raw_input("Enter sentence: ")
		self.classifier = raw_input("Enter type of classifier (NB or KNN or SVM): ")
		if self.classifier == "NB":
			self.naiveBayesClassifier()
		elif self.classifier == "KNN":
			self.KNNClassifier()
		elif self.classifier == "SVM":
			self.SVMClassifier()
		XInp = self.vectorizer.transform([self.inp])
		# self.XInpTfIdf = self.tfidfTransformer.transform(XInp)
		self.predicted = self.clf.predict(XInp)
		print(self.docs[int(self.predicted)][:-4])
	
if __name__ == '__main__':
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		training_docs = os.listdir(os.getcwd() + '/Data/')
		cf = Classifier(training_docs)
		cf.setTestAndTrainingData()
		cf.transformFeatures()
		inp = raw_input("Enter P for input prediction OR T for testing model: ")
		if inp == "P":
			cf.inputPrediction()
		else:
			cf.testModel()
