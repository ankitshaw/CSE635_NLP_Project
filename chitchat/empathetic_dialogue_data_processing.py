import pandas as pd
import json
import os

# print(os.listdir("./"))
# print(os.listdir("../datasets/empatheticdialogues"))
df = open("../datasets/empatheticdialogues/test.csv", encoding="utf8").readlines()
# df = pd.read_json("../datasets/chitchat.json",orient = "index")
# df = df[['prompt']]
# print(df.head())
# df.to_csv("data.csv")
inn=[]
out=[]
i = 0
while(i<len(df)):
    cparts = df[i - 1].strip().split(",")
    sparts = df[i].strip().split(",")
    if cparts[0] == sparts[0]:
        sent = cparts[5].replace("_comma_", ",")
        recv = sparts[5].replace("_comma_",",")
        inn.append(sent)
        out.append(recv)
        i+=2
    else:
        i+=1
df = pd.DataFrame({'send':inn,'recv':out})
df.to_csv("emp_data.csv")
