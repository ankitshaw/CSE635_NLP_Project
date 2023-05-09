
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load pre-trained sentence transformer model
sentence_transformer_model = "albert-base-v2"
sentence_transformer = SentenceTransformer(sentence_transformer_model)

def find_synonyms(keywords):
    synonyms = []
    for keyword in keywords:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
    return synonyms

def classify_feedback(text, handle_sentiment=True):
    sentiment = 0
    if handle_sentiment:
        sentiment = sid.polarity_scores(text)['compound']
        if sentiment < -0.5:
            return [3, sentiment, sentiment] # change topic
    sentence_embedding = sentence_transformer.encode([text])[0]
    exit_keywords = ['exit', 'quit', 'bye', 'goodbye', 'see you', 'see ya', 'later', "talk later", "gtg", "gotta go", "cya", "end chat", "end conversation"]
    exit_synonyms = find_synonyms(exit_keywords)
    exit_similarity = max([cosine_similarity([sentence_embedding], [sentence_transformer.encode([synonym])[0]])[0][0] for synonym in exit_keywords + exit_synonyms])
    change_topic_keywords = ["yeah", "okay", "uh huh", "I don't want to talk about this", "change", "topic", "change topic", 'new topic', 'different topic', 'hmm']
    change_topic_synonyms = find_synonyms(change_topic_keywords)
    print(exit_similarity)
    change_topic_similarity = max([cosine_similarity([sentence_embedding], [sentence_transformer.encode([synonym])[0]])[0][0] for synonym in change_topic_keywords + change_topic_synonyms])
    print(change_topic_similarity)

    if change_topic_similarity > 0.8:
        return [2, change_topic_similarity, sentiment]  # change topic
    elif exit_similarity > 0.8:
        return [1, exit_similarity, sentiment]  # exit chat
    else:
        return [0, probs[1], sentiment]  # continue chat
