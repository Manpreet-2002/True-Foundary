## True-Foundary
# Sentiment Analysis Binary Classification


(Code is available in the ```Model``` folder in ```venv```)


https://drive.google.com/file/d/1RC-ipqjXt2k3uHH-oyEb_VAl7Zdm5f9U/view?usp=sharing



Technologies used- Fast API for building the API server, Postman for testing the API, ML libraries used were ```scikit learn``` and ```tensorflow```.


Experimented with various algorithms like logistic regression, Naive Bayes classifier, Support Vector Machine and Word2vec. SVM had the highest accuracy (92%) and SVM model was built.


The Fast API server takes in a JSON data object with a message key, on the ‘/predict’ endpoint and returns the sentiment associated with the message’s value. 


The sentiment can have two possible values - positive or negative. After running the algorithm, the API sends a JSON resposne with the sentiment key and value.



# Sentiment Analysis API
1) -> Preprocessing of textual data
Casing -> converting to lower case
Denosiing -> removing URLs, ‘#’, ‘@’, RT, punctuations, and emojis
Tokenization and stop words removal
Stemming
Lemmatization

2) -> Test train split
Used sklearn library to split the training data and testing data and the ratio to split was 1:0.3 between train and test data.

3) -> Tf-Idf vectorizer
Fitted the object of class TfidfVectorizer() with the train data and saved it in a pickle file.
Tf-Idf vectorizer was used because it is a better approach than the count vectorizer and it is also widely used in NLP applications.

4) -> Training of different models
Experimented with different models and found out the Support Vector Machine had the highest accuracy.
Fitted the SVM model with the training data and saved it as a pickle file.
Set up of a FAST API server was done which accepts a POST request on the ’predict’ endpoint.
Performed preprocessing of the message field in the JSON data coming from the POST request and predicted the sentiment of the message.


Summary of different models experimented with:-

Logistic Regression  -> 91.077 %


Naive Bayes classifier -> 84.319 %


Support Vector Machine -> 92.001 %


Text embeddings -> 15.82%

