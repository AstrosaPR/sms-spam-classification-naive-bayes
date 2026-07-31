import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk

message = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
message['length'] = message['message'].apply(len)

def text_process(mess: str):
    import string
    mess = mess.lower()
    nopuct = [char for char in mess if char not in string.punctuation]
    nopuct = ''.join(nopuct)
    return [word for word in nopuct.split() if word.lower() not in stopwords.words('english')]


bow = CountVectorizer(analyzer=text_process).fit(message['message'])
message_bow = bow.transform(message['message'])

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(message_bow)
message_tfidf = tfidf_transformer.transform(message_bow)

from sklearn.naive_bayes import MultinomialNB

spam_detector = MultinomialNB().fit(message_tfidf, message['label'])

all_prediction = spam_detector.predict(message_tfidf)

from sklearn.metrics import classification_report

# print(classification_report(message['label'], all_prediction))

from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(message['message'],
#                                                     message['label'],
#                                                     test_size=0.2, random_state=42)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

pipeline.fit(message['message'], message['label'])


# pipline.fit(x_train, y_train)
# prediction = pipline.predict(x_test)
# print(classification_report(y_test, prediction))
# Singleton Prediction

def singleton_prediction(text_message: str):
    prediction = pipeline.predict([text_message])
    probability = pipeline.predict_proba([text_message])
    if prediction[0] == 'spam':
        result = 'spam'
    else:
        result = 'ham'

    print(
        f"Text: {text_message} \n\t Prediction: {result}\n\t Probability: Ham: {probability[0][0]:.2f}, Spam: {probability[0][1]:.2f}")


sample1_text = "CONGRATULATIONS! You won a free iPhone. Click here to claim your prize now!"
sample2_text = "Hey, are we still meeting for dinner at 8pm?"
sample3_text = "URGENT: Your bank account has been suspended. Verify your details immediately at http://fakebank.com"
sample4_text = "WINNER!! Free prize money URGENT claim now"

singleton_prediction(sample1_text)
print('-' * 100)
singleton_prediction(sample2_text)
print('-' * 100)
singleton_prediction(sample3_text)
print('-' * 100)
singleton_prediction(sample4_text)
# -------------------------------------------------------------
# Deploy Model
# joblib

# import joblib
# joblib.dump(spam_detector, 'spam_detector.joblib', compress=3)
# model =  joblib.load('spam_detector.joblib')
#
# import pickle
#
# with open('spam_detector.pkl', 'wb') as model_file:
#     pickle.dump(pipeline, model_file)


# site_url = '''https://sematec-co.com/wp-content/uploads/elementor/thumbs/%D8%AF%D9%88%D8%B1%D9%87-%D8%AC%D8%A7%D9%85%D8%B9-%D8%A8%D8%B1%D9%86%D8%A7%D9%85%D9%87%E2%80%8C%D9%86%D9%88%DB%8C%D8%B3%DB%8C-%D8%A8%D8%A7-%D9%BE%D8%A7%DB%8C%D8%AA%D9%88%D9%86-1-rm1plmu3s7tzb3sy5klta3p8r0qh2cpqqcckwbvfao.jpg'''
# import requests
#
# response = requests.get(site_url)
# print(response.content)
#
# with open("site.jpg", "wb") as f:
#     f.write(response.content)


