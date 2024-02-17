import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import nltk
import re
from nltk.corpus import stopwords
import string 
from joblib import dump

# import dataset
data= pd.read_csv('twitter_data.csv')

# Process the dataset
data['labels']=data['class'].map({0:'Hate Speech',1:'Offensive Speech',2:'No Hate and Offensive Speech'})
data=data[['tweet','labels']]

stopwords= set(stopwords.words('english'))
stemmer=nltk.SnowballStemmer('english')

def clean(text):
    text=str(text).lower()
    text=re.sub('[.?]','',text)
    text=re.sub('https?://\S+|www.\S+', '',text)
    text=re.sub('<.?>+','',text)
    text=re.sub(r'[^\w\s]','',text)
    text=re.sub('\n','',text)
    text=re.sub('\w\d\w','',text)
    text=[word for word in text.split(' ')if word not in stopwords]
    text=' '.join(text)
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=' '.join(text)
    return text
data['tweet']=data['tweet'].apply(clean)

# Vectorize the data
x=np.array(data['tweet'])
y=np.array(data['labels'])
cv=CountVectorizer()
x=cv.fit_transform(x)

# Spliting dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30,random_state=42)

# Model Train
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

# Saving model and vectorizing
dump(model,'hate_speech_model.pkl')
dump(cv,'count_vectorizer.pkl')

from joblib import load
model=load('hate_speech_model.pkl')
cv=load('count_vectorizer.pkl')

# function process and predict
def predict_speech(text):
    text=clean(text)
    text_vectorized=cv.transform([text]).toarray()
    prediction=model.predict(text_vectorized)
    return prediction[0]

# use streamlit for web
import streamlit as st 
from joblib import load

model=load('hate_speech_model.pkl')
cv=load('count_vectorizer.pkl')

def hate_speech(tweet):
    data = cv.transform([tweet]).toarray()
    prediction=model.predict(data)
    return prediction[0]

st.title("Hate Speech Detection")
user_input = st.text_area("Enter a Tweet:")

if user_input:
    prediction = hate_speech(user_input)
    st.write(f"Prediction: {prediction}")
