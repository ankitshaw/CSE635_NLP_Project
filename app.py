import streamlit as st
from streamlit_chat import message
from chitchat.chitchat import chitchat
import intent_classifier
from nlp_pipeline import get_nel
from wikibot.wiki_ir import TopicBot

topicBot = TopicBot()

if 'response' not in st.session_state:
    st.session_state['response'] = []
if 'input' not in st.session_state:
    st.session_state['input'] = []

def get_response(prompt):
    #Call the dialog manager api to get the response for the prompt
    # message=prompt + ": Reply"
    #dialogue manager rule or model get the prompt and call individual generator
    links = get_nel(prompt)
    if intent_classifier.classify(prompt=prompt, topics=len(links)) == "chitchat":
        print("doing chitchat")
        message = chitchat(prompt) #directly calling chitchat for testing
    else:
        #perform entity recoq, linker, find relevant facts, perform paraphrasing and return
        # return message
        print("doing wiki")
        message = topicBot.generator(prompt, links)
        if message == "":
            message = chitchat(prompt)

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
