from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

exit_keywords = ['exit', 'quit', 'bye', 'goodbye', 'see you', 'see ya', 'later', "talk later", "gtg", "gotta go", "cya", "end chat", "end conversation"]
change_topic_keywords = ["yeah", "okay", "uh huh", "I don't want to talk about this", "change", "topic", "change topic", 'new topic', 'different topic', 'hmm']

model = SentenceTransformer('all-MiniLM-L6-v2')

exit_embeddings = model.encode(exit_keywords, convert_to_tensor=True)
topic_embeddings = model.encode(change_topic_keywords, convert_to_tensor=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stop words
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    
    # Lemmatize words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

def check_similarity(input, target_embeddings):
    # Preprocess input
    input = preprocess_text(input)
    
    # Compute embeddings
    input_embeddings = model.encode(input, convert_to_tensor=True)

    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(input_embeddings, target_embeddings)
    return float(max(cosine_scores[0])), exit_keywords[np.argmax(cosine_scores[0])]

def check_feedback(input):
    # change topic check
    change_topic_similarity = check_similarity(input, topic_embeddings)
    exit_similarity = check_similarity(input, exit_embeddings)
    if change_topic_similarity[0] > 0.6:
        return (2, change_topic_similarity)
    elif exit_similarity[0] > 0.6:
        return (1, exit_similarity)  # exit chat
    else:
        return (0, 0)  # continue chat
