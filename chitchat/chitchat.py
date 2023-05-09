from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname).to(device)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

def chitchat(prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    reply_ids = model.generate(**inputs)
    print(tokenizer.batch_decode(reply_ids)[0].replace("<s>","").replace("</s>","").strip())
    return tokenizer.batch_decode(reply_ids)[0].replace("<s>","").replace("</s>","").strip()

def chitchat_batch(prompt_list):
    # print(prompt_list)
    inputs = tokenizer(prompt_list, return_tensors="pt", truncation=True,padding=True).to(device)
    reply_ids = model.generate(**inputs)
    # print(tokenizer.batch_decode(reply_ids))
    # print(tokenizer.batch_decode(reply_ids)[0].replace("<s>","").replace("</s>","").strip())
    return tokenizer.batch_decode(reply_ids)
