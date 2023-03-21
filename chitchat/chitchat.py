import chitchat_dataset as ccc

dataset = ccc.Dataset()

# Dataset is a subclass of dict()
for convo_id, convo in dataset.items():
    print(convo_id, convo)