import pandas as pd
import json

df = pd.read_json("../datasets/chitchat.json",orient = "index")
# df = df[['prompt']]
# print(df.head())
# df.to_csv("data.csv")
prompt = []
ind = []
out = []
for index,row in df.iterrows():
    # print(index)
    ind.append(index)
    messages = row['messages']
    # print(messages)
    msg1 = messages[0]
    temp = ""
    for text in msg1:
        temp = temp + " " + text['text']
    prompt.append(temp)
    
    temp = ""
    msg2 = None
    if len(messages) >= 2:
        msg2 = messages[1]
    
    if msg2:
        for text in msg2:
            temp = text['text'] + " " + temp
    out.append(temp)
    # print(indexprompt[-1],out[-1])
    # break
print(len(prompt),len(out), len(ind))

df = pd.DataFrame({'id':ind,'in':prompt,'out':out})
df.to_csv("data2.csv")