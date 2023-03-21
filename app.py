import streamlit as st
from streamlit_chat import message
import os 


def get_response(prompt):
    message=prompt + ": Reply"
    return message

st.title("Chat Bot App")
user_input=st.text_input("You:",key='input')
if user_input:
    output=get_response(user_input)

    st.session_state['input'].append(user_input)
    st.session_state['response'].append(output)
if st.session_state['response']:
    for i in range(len(st.session_state['response'])-1, -1, -1):
        message(st.session_state["response"][i], key=str(i))
        message(st.session_state['input'][i], is_user=True, key=str(i) + '_user')
