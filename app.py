import streamlit as st
from streamlit_chat import message
from dialog_manager import model_selection
import logging
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image

image = Image.open('img-removebg-preview.png')

st.image(image,width=300)

# topicBot = TopicBot()

if 'response' not in st.session_state:
    st.session_state['response'] = []
if 'input' not in st.session_state:
    st.session_state['input'] = []

def get_response(prompt):
    #Call the dialog manager api to get the response for the prompt
    # message=prompt + ": Reply"
    #dialogue manager rule or model get the prompt and call individual generator
    # links = get_nel(prompt)
    response = model_selection(prompt)
    chat_message = response[0]
    history = response[0]
    return chat_message

st.title("BabbleGo")
user_input=st.text_input("You:",key='user')

if user_input:
    tryy = 1
    # output = ""
    # try:
    output=get_response(user_input)
    # except:
        # output="I didn't got that. I am sorry"  
    st.session_state['input'].append(user_input)
    st.session_state['response'].append(output)

if st.session_state['response']:
    for i in range(len(st.session_state['response'])-1, -1, -1):
        message(st.session_state["response"][i], key=str(i))
        message(st.session_state['input'][i], is_user=True, key=str(i) + '_user')