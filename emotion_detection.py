from transformers import pipeline, set_seed
import time
import torch

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the pipeline with batch_size=8
classifier = pipeline("text-classification", model='bhadresh-savani/albert-base-v2-emotion', return_all_scores=True, device=device, batch_size=1)

# Set a seed for reproducibility
set_seed(42)

# # Define a cache for storing the output of the model
# cache = {}

# Define a function to get the prediction for a single input
def predict_emotion(input_text):
    prediction = classifier(input_text)
    # cache[input_text] = prediction
    return prediction

# Test the predict function for each input text and measure the time taken for each prediction
def detect_emotion(input_text):
    # start_time = time.time()
    print(f"Input: {input_text}")
    prediction = predict_emotion(input_text)
    # print(f"Output: {prediction}")
    max_score_label_dict = max(prediction[0], key=lambda x:x['score'])
    max_score_label = max_score_label_dict['label']
    max_score = max_score_label_dict['score']
    print(f"Label with maximum score: {max_score_label} (Score: {max_score:.4f})")
    return max_score_label, max_score
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time:.4f} seconds")