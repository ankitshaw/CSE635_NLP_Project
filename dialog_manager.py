from chitchat.chitchat import chitchat
import intent_classifier
from nlp_pipeline import get_nel
# from wikibot.wiki_ir import TopicBot
from wikibot.wikibot import get_wiki_response
from intent_classifier_albert import classify
import emotion_detection
from user_feedback_sbert import check_feedback
from  sentiment_prediction import get_sentiment
import torch

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def general_case(text, intent):
    if intent == "chitchat":
        print("doing chitchat")
        #directly calling chitchat for testing
        message = chitchat(text)
    else:
        print("doing wiki")
        #directly calling information reterival for testing
        message = get_wiki_response(text)
    return message

def model_selection(prompt):
    #Call the dialog manager api to get the response for the prompt
    # message=prompt + ": Reply"
    #dialogue manager rule or model get the prompt and call individual generator
    """
    Cases for DM:
    1. Emotion 1 — Intent is IR — Explicit Feedback 0. —> Chitchat —> IR
    2. Emotion 1 — Intent is chitchat — Explicit Feedback 0. —> Chitchat
    3. Emotion 0 — Intent is IR — Explicit Feedback 0. —> IR
    4. Emotion 1/0 — Intent is IR — Explicit Feedback 1 (exit)—> “Thank You”
    5. Emotion 1/0 — Intent is IR — Explicit Feedback 1 (topic change)/ Implicit Feedback(User not that interested or engaged)—> “What do you want to talk about next?”
    6. Example of user input topic “Solar system”, “Movie”: — > (test both) Chitchat / intent is IR —> IR response(OR)
    7. We suggest user any top 5 topics to talk about:
    """

    intent_type, emotion_ts, feedback_recv = None, None, None
    emotion_type, emotion_score  = None, 0
    message = ""
    history = 0
    try:
        # Code that may raise an exception
        intent_type = classify(prompt)
        print("intent_type:", intent_type)
    except:
        # Handle the exception
        print("intent not detected")
    # try:
    #     # Code that may raise an exception
    #     emotion_ts = emotion_detection.detect_emotion(prompt)
    #     if emotion_ts is not None:
    #         emotion_type = emotion_ts[0]
    #         emotion_score = emotion_ts[1]
    # except:
    #     # Handle the exception
    #     print("emotion not detected")
    try:
        # Code that may raise an exception
        feedback_recv = check_feedback(prompt)
        print("feedback_recv:", feedback_recv)
    except:
        # Handle the exception
        print("feedback not detected")
    try:
        # Code that may raise an exception
        sentiment_recv = get_sentiment(prompt)
        print("sentiment_recv:", sentiment_recv)
    except:
        # Handle the exception
        print("sentiment not detected")

    if feedback_recv is not None:
        if feedback_recv[0] == 1:
            message = "Thank You for using our service. Ping me if you need any help"
            return message, history
        elif feedback_recv[0] == 2:
            message = "Please let me know, what you want to talk about next?"
            return message, history
        else:
            if sentiment_recv is not None:
                if sentiment_recv > 0.8 or sentiment_recv < -0.5:
                    message = chitchat(prompt)
                    history = 1
                else:
                    message = general_case(prompt, intent_type)
                    history = 0
                return message, history
            else:
                message = general_case(prompt, intent_type)
                return message, history
    else:
        message = general_case(prompt, intent_type)
        return message, history
