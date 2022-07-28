from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class Tdidf_vectorizer():
    def __init__(self, _message):
        self.message = [_message]
    def vectorizer(self):
        vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
        self.message = vectorizer.transform(self.message)
        return self.message