# This is our project: "HATE SPEECH DETECTION"

- The basic concept to making this project to detect the message or comment of any person whether their message is hate, or offensive or neither hate or offensive.

- In this project we use python library and machine learning module to train the data

- We use numpy, Pandas, Scikit-learn, nltk modules,decision tree classifier and stopwords. 
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
- We import the data and process the dataset and split the dataset and train the model and predict the result.
- We are using Streamlit to deploy ML module into web page.
.import streamlit as st 