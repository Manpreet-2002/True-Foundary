# True-Foundary
Sentiment Analysis Binary Classification
Sentiment Analysis API
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

