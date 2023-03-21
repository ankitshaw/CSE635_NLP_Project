from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

def chitchat(prompt):
    inputs = tokenizer([prompt], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    print(tokenizer.batch_decode(reply_ids)[0].replace("<s>","").replace("</s>","").strip())
    return tokenizer.batch_decode(reply_ids)[0].replace("<s>","").replace("</s>","").strip()