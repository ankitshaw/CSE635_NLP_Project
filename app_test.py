# import streamlit as st
# from streamlit_chat import message
from chitchat.chitchat import chitchat, chitchat_batch
# import intent_classifier
# from nlp_pipeline import get_nel
from tqdm import tqdm
# from wikibot.wiki_ir import TopicBot
# from wikibot.wikibot import get_wiki_response
# import intent_classifier_albert
# from torch.utils.data import Dataset, DataLoader
import evaluate

from PIL import Image

# image = Image.open('img-removebg-preview.png')

# st.image(image,width=300)
def batch(iterable, n=20):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_response(prompt):
    #Call the dialog manager api to get the response for the prompt
    # message=prompt + ": Reply"
    #dialogue manager rule or model get the prompt and call individual generator
    # links = get_nel(prompt)
    # if inference_intent_classifier_trained_albert.classify(prompt) == "chitchat":
        # print("doing chitchat")
    message = chitchat_batch(prompt) #directly calling chitchat for testing
    # else:
        #perform entity recoq, linker, find relevant facts, perform paraphrasing and return
        # print("doing wiki")
        # message = topicBot.generator(prompt, links)
        # message = get_wiki_response(prompt)
        # if message == "":
        #     message = chitchat(prompt)
    return message

# def get_intent():


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
    df = pd.read_csv("chitchat/emp_data.csv")
    # data = CustomDataset(df)
    # loader = DataLoader(data,batch_size=100)
    # return
    bot_out = []#["" for i in range(df.size)]
    # print(len(bot_out))

    for btch in tqdm(batch(df['send'].tolist())):
    # for index, row in tqdm(df.iterrows()):
        # print(btch)
        count += 1
        out = []
        # try:
        print(len(btch))
        out = get_response(btch)
        # except:
        #     out=["I didn't got that. I am sorry"]
       
        # print(out)
        # print(count)
        # df['bot_out']= out
        # print(out)
        bot_out.extend(out)
        # count += 1
    df['bot_out']=bot_out
    df.to_csv("emp_data_out.csv")

def bleu_inference_cc():
    import pandas as pd
    count = 0
    df = pd.read_csv("chitchat/data_chitchat.csv")
    # data = CustomDataset(df)
    # loader = DataLoader(data,batch_size=100)
    # return
    bot_out = []#["" for i in range(df.size)]
    # print(len(bot_out))

    for btch in tqdm(batch(df['in'].tolist())):
    # for index, row in tqdm(df.iterrows()):
        # print(btch)
        count += 1
        out = []
        # try:
        print(len(btch))
        out = get_response(btch)
        # except:
        #     out=["I didn't got that. I am sorry"]
       
        # print(out)
        # print(count)
        # df['bot_out']= out
        # print(out)
        bot_out.extend(out)
        # count += 1
    df['bot_out']=bot_out
    df.to_csv("cc_data_out.csv")

def evaluator(metric, dataset):
    import pandas as pd
    bleu = evaluate.load(metric)
    df = pd.read_csv(dataset)
    bot_out = []
    out = []
    for index, row in df.iterrows():
        if not pd.isnull(row['bot_out']) and not pd.isnull(row['recv']):
            out.append(row['recv'].strip())
            bot_out.append(row['bot_out'].strip())
    results = bleu.compute(predictions=bot_out,references=out)
    print(results)

def bert_score(metric, dataset):
    import pandas as pd
    bleu = evaluate.load(metric)
    df = pd.read_csv(dataset)
    bot_out = []
    out = []
    for index, row in df.iterrows():
        if not pd.isnull(row['bot_out']) and not pd.isnull(row['recv']):
            out.append(row['recv'].strip())
            bot_out.append(row['bot_out'].strip())
    results = bleu.compute(predictions=bot_out,references=out,lang="en")
    print(results)

def bleurt_score(metric, dataset):
    import pandas as pd
    bleu = evaluate.load(metric, module_type="metric")
    df = pd.read_csv(dataset)
    bot_out = []
    out = []
    for index, row in df.iterrows():
        if not pd.isnull(row['bot_out']) and not pd.isnull(row['recv']):
            out.append(row['recv'].strip())
            bot_out.append(row['bot_out'].strip())
    results = bleu.compute(predictions=bot_out,references=out)
    print(results)

def perplexity(metric, dataset):
    import pandas as pd
    bleu = evaluate.load(metric,module_type="metric")
    df = pd.read_csv(dataset)
    bot_out = []
    out = []
    for index, row in df.iterrows():
        if not pd.isnull(row['bot_out']) and not pd.isnull(row['recv']):
            out.append(row['recv'].strip())
            bot_out.append(row['bot_out'].strip())
    results = bleu.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=bot_out)
    print(results)


# evaluator("bleu","chitchat/emp_data_out.csv")
# bert_score("bertscore","chitchat/emp_data_out.csv")
# evaluator("rouge","chitchat/emp_data_out.csv")
# bleurt_score("bleurt","chitchat/emp_data_out.csv")
# perplexity("perplexity","chitchat/emp_data_out.csv")
bleu_inference_cc()