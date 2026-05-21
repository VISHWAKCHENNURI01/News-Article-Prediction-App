import streamlit as st
import pickle

# Load model
with open("news_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("News Article Prediction App")

# Description
st.write("Enter news article text below")

# Text input
user_input = st.text_area("Enter News Text")

# Predict button
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:
        prediction = model.predict([user_input])

        st.success(f"Predicted Category: {prediction[0]}")