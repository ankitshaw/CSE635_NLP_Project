import pandas as pd
import json

df = pd.read_json("../datasets/chitchat.json",orient = "index")
# df = df[['prompt']]
# print(df.head())
# df.to_csv("data.csv")
prompt = []
ind = []
for index,row in df.iterrows():
    # print(index)
    ind.append(index)
    prompt.append(row['prompt'])
    messages = row['messages']
    # print(messages)
    for msg in messages:
        for text in msg:
            ind.append(text['sender'])
            prompt.append(text['text'])
print(len(prompt),len(ind))

df = pd.DataFrame({'parent_id':ind,'parent':prompt})
df.to_csv("data.csv")