##importing libraries

##data manipulation
import re
import string

##methods and stopwords text preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from emot.emo_unicode import UNICODE_EMOJI # For emojis

##Creating stopwords set
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


UNICODE_EMO = UNICODE_EMOJI
class Preprocess_Message():
    def __init__(self, _message):
        self.message = _message
    # Function for converting emojis into word
    def convert_emojis(self):
        for emot in UNICODE_EMO:
            self.message = self.message.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
        return self.message
    def preprocess_message(self):
        """
        Runs a set of transformational steps to preprocess
        the text of the message
        """
        #Casing -> convert text to lowercase
        msg = self.message
        self.message = msg.lower()
        
        #Denoising -> remove urls
        self.message = re.sub(r"http\S+|www\S+|https\S+", "", self.message, flags=re.MULTILINE)
        
        #Denoising -> remove emojis
        self.message = self.convert_emojis()
        
        #Denoising -> remove punctuations
        self.message = self.message.translate(str.maketrans("", "", string.punctuation))
        
        #Denoising -> remove @ and # from message
        self.message = re.sub(r"@[a-z0-9]+", "", self.message)
        self.message = re.sub(r"#", "", self.message)
        
        #Denoising -> remove RT
        self.message = re.sub(r"RT[\s]+", "", self.message)
        
        #Tokenization and stop words removal
        message_tokens = word_tokenize(self.message)
        filtered_words = [word for word in message_tokens if word not in stop_words]
        
        #Text normalization -> stemming
        ps = PorterStemmer()
        stemmed_words = [ps.stem(w) for w in filtered_words]
        
        #Text normalization -> lemmatization
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
        
        return " ".join(lemma_words)