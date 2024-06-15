import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

#lets load the saved vectorizer and naive bayes
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

#transform  text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text=text.lower() #converting to lower
    text=nltk.word_tokenize(text)#Tokenize

    #Removing special characters & retaining alphanumeric
    text=[word for word in text if word.isalnum()]

    #Removing stopwords & Punctuation
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    #Apply stemming
    text=[ps.stem(word) for word in text]

    return " ".join(text)

#saving streamlit code
st.title("Email Spam Classifier")
input_sms=st.text_area("Enter Message")

if st.button('Predict'):
    #preprocess
    transformed_sms=transform_text(input_sms)
    #vectorize
    vector_input=tfidf.transform([transformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    #Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")