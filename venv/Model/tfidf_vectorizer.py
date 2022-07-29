from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from preprocess import Preprocess_Message

class Tdidf_vectorizer():
    def __init__(self, _message):
        msg_obj = Preprocess_Message(_message)
        msg = msg_obj.preprocess_message()
        self.message = [msg]
        print(self.message)
    def vectorizer(self):
        vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
        self.message = vectorizer.transform(self.message)
        return self.message

