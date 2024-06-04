import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import ImageTk, Image
from tkinter import Label
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from tkinter import Label, Button, Entry, Tk, messagebox

from sklearn.tree import DecisionTreeClassifier
# Loading the saved model
with open('XGBoost.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
#from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('phishing.csv')
X = df.Domain
y = df.Label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#model = Pipeline([('tfidf', TfidfVectorizer()), ('cls', LinearSVC())])
model = Pipeline([('tfidf', TfidfVectorizer()), ('cls', DecisionTreeClassifier())])
model.fit(X_train, y_train)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Assuming X_train is your training text data

# Define the model pipeline (replace this with your actual model setup)
model = Pipeline([('tfidf', tfidf_vectorizer), ('cls', DecisionTreeClassifier())])
model.fit(X_train, y_train)  # Assuming y_train is your training labels


def predict(input_text):
    processed_text = tfidf_vectorizer.transform([input_text])
    prediction = loaded_model.predict(processed_text)
    return prediction[0]

print("Welcome to Phishing Detection System")

while True:
    user_input = input("Enter a domain name to check for phishing (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Thank you for using Phishing Detection System")
        break
    else:
        result = predict(user_input)
        if result == 0:
            print(f"The Entered Domain '{user_input}' Is Classified As Website Is Safe.")
        else:
            print(f"The Entered Domain '{user_input}' Is Classified As Website Is Unsafe.")
