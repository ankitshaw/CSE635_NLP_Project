from transformers import AutoTokenizer, AlbertForSequenceClassification

class AlbertModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.model_albert = AlbertForSequenceClassification.from_pretrained('albert-base-v2')