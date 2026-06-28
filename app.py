import streamlit as st 
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

st.set_page_config(page_title="Spam Detector", layout="centered")

ps = PorterStemmer()

vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Spam Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a message and check whether it's Spam or Not</p>", unsafe_allow_html=True)

input_sms = st.text_area(" Enter your message here:", height=150)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button(" Predict")
#  NEW CLEAR BUTTON
with col3:
    clear_btn = st.button(" Clear")

# TEXT PROCESSING
def transform_text(text):
    text = text.lower()
    text = nltk.wordpunct_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # REMOVE STEMMING (important)
    return " ".join(y)

# PREDICTION
if predict_btn:
    if input_sms.strip() != "":
        transform_sms = transform_text(input_sms)

        vector_input = vectorizer.transform([transform_sms])

        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success(" Not Spam")

    else:
        st.warning("Please enter a message")

# clear button logic
if clear_btn:
    st.session_state.text = ""
    st.rerun()
