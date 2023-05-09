from chitchat.chitchat import chitchat
import intent_classifier
from nlp_pipeline import get_nel
# from wikibot.wiki_ir import TopicBot
from wikibot.wikibot import get_wiki_response
import inference_intent_classifier_trained_albert


def model_selection(prompt):
    #Call the dialog manager api to get the response for the prompt
    # message=prompt + ": Reply"
    #dialogue manager rule or model get the prompt and call individual generator
    # links = get_nel(prompt)
    if inference_intent_classifier_trained_albert.classify(prompt) == "chitchat":
        print("doing chitchat")
        message = chitchat(prompt) #directly calling chitchat for testing
    else:
        #perform entity recoq, linker, find relevant facts, perform paraphrasing and return
        print("doing wiki")
        # message = topicBot.generator(prompt, links)
        message = get_wiki_response(prompt)
        # if message == "":
        #     message = chitchat(prompt)

    return message
