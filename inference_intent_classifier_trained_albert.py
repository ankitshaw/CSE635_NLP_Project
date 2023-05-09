# -*- coding: utf-8 -*-
from transformers import AlbertForSequenceClassification, AdamW, AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import numpy as np
import re

# def __init__():
path_to_model = 'models/albert_model_intent_classification.pt'

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2').to(device)
loaded_model = None

def preprocess(input_text):
  # Tokenize the input text and convert to PyTorch tensors
  # my_array = np.array(input_text)
  test_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt').to(device)
  # Create a PyTorch dataset and dataloader
  input_data = TensorDataset(test_input['input_ids'], test_input['attention_mask'])
  test_dataloader = DataLoader(input_data)
  return test_dataloader

def load_trained_model(model_path, model):
  print("Loading the weights of the model...")
  
  if torch.cuda.is_available():
    device = torch.device("cuda")
    trained_model = model.load_state_dict(torch.load(model_path))
  else:
    device = torch.device("cpu")
    trained_model = model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
  trained_model.to(device)
  return trained_model

def load_trained_model(model_path, model, device):
  print("Loading the weights of the model...")
  
  loaded_state_dict = torch.load(model_path, map_location=device)
  new_model = model
  new_model.load_state_dict(loaded_state_dict)
  new_model.to(device)
  
  return new_model

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader):
    net.eval()
    with torch.no_grad():
        for input_ids, attn_mask in tqdm(dataloader):
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            outputs = net(input_ids, attn_mask)
            logits = outputs.logits  # extract the logits from the output
            prob = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
    return prob

def prob_to_class(x):
  x = x.tolist()[0]
  class_x = x.index(max(x))
  return class_x


def classify(user_input):
    global loaded_model
    user_input = re.sub(r'[^\w\s]+', '', user_input)
    print("user_input: ",user_input)
    if loaded_model == None:
       loaded_model = load_trained_model(path_to_model, model, device)
    print("Predicting on test data...")
    prediction_prob = test_prediction(net=loaded_model, device=device, dataloader=preprocess(user_input))
    class_predicted = prob_to_class(prediction_prob)
    print("predict class index: ", class_predicted)
    if class_predicted == 0:
        return "chitchat"
    else:
        return "topic"