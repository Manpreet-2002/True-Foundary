import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

##methods and stopwords text preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from emot.emo_unicode import UNICODE_EMOJI # For emojis
warnings.filterwarnings("ignore")

with open('model_pickle','rb') as f:
    model = pickle.load(f)

class Msg(BaseModel):
    message : str

##Creating stopwords set
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI()

UNICODE_EMO = UNICODE_EMOJI
# Function for converting emojis into word
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    return text

def preprocess_message(message):
    """
    Runs a set of transformational steps to preprocess
    the text of the message
    """
    
    #Casing -> convert text to lowercase
    message = message.lower()
    
    #Denoising -> remove urls
    message = re.sub(r"http\S+|www\S+|https\S+", "", message, flags=re.MULTILINE)
    
    #Denoising -> remove emojis
    message = convert_emojis(message)
    
    #Denoising -> remove punctuations
    message = message.translate(str.maketrans("", "", string.punctuation))
    
    #Denoising -> remove @ and # from message
    message = re.sub(r"@[a-z0-9]+", "", message)
    message = re.sub(r"#", "", message)
    
    #Denoising -> remove RT
    message = re.sub(r"RT[\s]+", "", message)
    
    #Tokenization and stop words removal
    message_tokens = word_tokenize(message)
    filtered_words = [word for word in message_tokens if word not in stop_words]
    
    #Text normalization -> stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    
    #Text normalization -> lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(lemma_words)

@app.get("/")
async def root():
    return {"Sentiment Analysis API setup successful!"}

@app.post("/predict")
async def predict_sentiment(data : Msg):
    data = data.dict()
    message = data['message']
    msg = preprocess_message(message)
    msg = [msg]
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    vectorized_message = vectorizer.transform(msg)
    sentiment = model.predict(vectorized_message)[0]
    if sentiment == 1:
        return {"sentiment" : "positive"}
    else:
        return {"sentiment" : "negative"}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)