import numpy as np
import pandas as pd
import streamlit as st
import pickle

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Message(SPAM or HAM) Classification')
df = pd.read_csv('Cleaned_spam.csv',usecols = ['Category','Cleaned'])
df

st.header('Enter Message')

message = st.text_area("Enter message:")

if st.button("Submit"):
	if message != "":
		message_data = {'predict_message':[message]}
		message_data_df = pd.DataFrame(message_data)
		prediction = model.predict(message_data_df['predict_message'])
		st.write("Predicted message category = ",prediction[0])
	else:
		st.write("Please Enter Message")

	if prediction[0] == 1:
		st.error("CAREFUL!!!, THIS MIGHT BE A SPAM MESSAGE")     
	else:
		st.success("THIS IS A HAM(NOT SPAM) MESSAGE")
    	          
st.sidebar.subheader("About App")
st.sidebar.info("This web app helps you to find out whether the message is spam or not")
st.sidebar.info("Enter the message and click on the 'Submit' button to check whether the message is either Spam or Ham")