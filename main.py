import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# step 1: decoders/encoders
word_index = imdb.get_word_index()
reversed_word_index = {v : k for k, v in word_index.items()}

# step 2: - load model
model = load_model('./simple_rnn_imdb.h5')

## step 3: - helper functions
# decoding review
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

# preprocess input
def preprocess_input(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# step 4: prediction functions
def predict_sentiment(prediction):
    # prepro_review = preprocess_input(review)
    # prediction = model.predict(prepro_review)
    
    sentiment = 'Postive' if prediction[0][0] > 0.5 else 'Negative' 

    return sentiment, prediction[0][0]


## step 5: steamlit app
st.title("Simple RNN IMDB Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

# input
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)
    sentiment, score = predict_sentiment(prediction)

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write("Please enter a movie review.")


    