import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import numpy as np

model=load_model('Simple_rnn_model.keras')

word_index=imdb.get_word_index()
index_with_word={value:key for key,value in word_index.items()}

def decoder(review):
    return ' '.join([index_with_word.get(i-3,'?') for i in review])

def process(text):
    words=text.lower().split()
    encoded=[word_index.get(word,2)+3 for word in words]
    padded=sequence.pad_sequences([encoded],maxlen=500)

    return padded

import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify is as postive and negative")

user_input=st.text_area('Movie Review')

if st.button('Classify'):
    processed_input=process(user_input)

    prediction=model.predict(processed_input)

    sentiment= "Postive" if prediction[0][0] > 0.5 else "Negative"

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")

else:
    st.write('Please Enter a movie review')