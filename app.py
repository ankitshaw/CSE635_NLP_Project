import streamlit as st
from streamlit_chat import message

if 'response' not in st.session_state:
    st.session_state['response'] = []
if 'input' not in st.session_state:
    st.session_state['input'] = []

def get_response(prompt):
    #Call the dialog manager api to get the response for the prompt
    message=prompt + ": Reply"
    return message

st.title("Wannabe Bot")
user_input=st.text_input("You:",key='user')

if user_input:
    output=get_response(user_input)
    st.session_state['input'].append(user_input)
    st.session_state['response'].append(output)


if st.session_state['response']:
    for i in range(len(st.session_state['response'])-1, -1, -1):
        message(st.session_state["response"][i], key=str(i))
        message(st.session_state['input'][i], is_user=True, key=str(i) + '_user')
