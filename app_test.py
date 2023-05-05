import streamlit as st
from streamlit_chat import message
from chitchat.chitchat import chitchat
import intent_classifier
from nlp_pipeline import get_nel
from wikibot.wiki_ir import TopicBot
import evaluate

import json

topicBot = TopicBot()

def get_response(prompt):
    #Call the dialog manager api to get the response for the prompt
    # message=prompt + ": Reply"
    #dialogue manager rule or model get the prompt and call individual generator
    links = [] #get_nel(prompt)
    # if intent_classifier.classify(prompt=prompt, topics=len(links)) == "chitchat":
    # print("doing chitchat")
    message = chitchat(prompt) #directly calling chitchat for testing
    # else:
    #     #perform entity recoq, linker, find relevant facts, perform paraphrasing and return
    #     # return message
    #     print("doing wiki")
    #     message = topicBot.generator(prompt, links)
    #     if message == "":
    #         message = chitchat(prompt)

    return message

def processpara(paras):
    result ={}
    for para in paras:
        for qas in para['qas']:
            question = qas['question']
            print(question)
            output = ""
            try:
                output=get_response(question)
            except:
                output="I didn't got that. I am sorry"
            result[qas['id']] = output
    return result


def squad ():
    file = open("dev-v2.0.json")
    data = json.load(file)
    datas = data['data']
    result = {}

    for data in datas:
        for para in data['paragraphs']:
            for qas in para['qas']:
                question = qas['question']
                print(question)
                output = ""
                try:
                    output=get_response(question)
                except:
                    output="I didn't got that. I am sorry"
                result[qas['id']] = output

    with open('result.json', 'w') as fp:
        json.dump(result, fp)

def bleu_inference():
    import pandas as pd
    count = 0
    df = pd.read_csv("chitchat/data2.csv")
    bot_out = []
    for index, row in df.iterrows():
        count += 1
        if count%50 != 0:
            # print(count)
            bot_out.append("")
            continue
        try:
            out = get_response(row['in'])
        except:
            out="I didn't got that. I am sorry"
       
        # print(out)
        print(count)
        # df['bot_out']= out
        bot_out.append(out)
        # count += 1
    df = pd.DataFrame({'bot_out':bot_out})
    df.to_csv("bleu.csv")

def bleu():
    import pandas as pd
    bleu = evaluate.load("bleu")
    df = pd.read_csv("chitchat/data2.csv")
    bot_out = []
    out = []
    for index, row in df.iterrows():
        if not pd.isnull(row['bot_out']) and not pd.isnull(row['out']):
            out.append(row['out'])
            bot_out.append(row['bot_out'])
    print(len(bot_out))
    results = bleu.compute(predictions=bot_out,references=out)
    print(results)

bleu()