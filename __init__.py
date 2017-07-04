import warnings
import os

class Classifier:
	"""
	"""
	def __init__(self, docs):
		self.docs = docs
	
	# Implement the class here
	
if __name__ == '__main__':
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		training_docs = os.listdir(os.getcwd() + '/Data/')
		cf = Classifier(training_docs)
		# Add code for Prediction and Training
