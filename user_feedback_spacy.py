import numpy as np
import spacy
import fasttext

exit_keywords = ['exit', 'quit', 'bye', 'goodbye', 'see you', 'see ya', 'later', "talk later", "gtg", "gotta go", "cya", "end chat", "end conversation"]
change_topic_keywords = ["yeah", "okay", "uh huh", "I don't want to talk about this", "change", "topic", "change topic", 'new topic', 'different topic', 'hmm']

nlp = spacy.load('en_core_web_lg')
word_embedding_model = fasttext.load_model('cc.en.300.bin')

exit_embeddings = [word_embedding_model.get_word_vector(word) for word in exit_keywords]
topic_embeddings = [word_embedding_model.get_word_vector(word) for word in change_topic_keywords]

def preprocess_text(text):
    # Tokenize text using spaCy
    doc = nlp(text)

    # Remove stop words, punctuation, and non-alphabetic characters
    tokens = [token for token in doc if not token.is_stop and token.is_alpha]

    # Lemmatize words
    lemmas = [token.lemma_ for token in tokens]

    return ' '.join(lemmas)

def check_similarity(input, target_embeddings):
    # Preprocess input
    input = preprocess_text(input)
    
    # Compute embeddings
    input_embeddings = [word_embedding_model.get_word_vector(word) for word in input.split()]

    # Compute cosine-similarities for each word with each other word
    cosine_scores = []
    for word_emb in input_embeddings:
        word_scores = [np.dot(word_emb, target_emb) / (np.linalg.norm(word_emb) * np.linalg.norm(target_emb)) for target_emb in target_embeddings]
        cosine_scores.append(max(word_scores))

    return float(max(cosine_scores)), exit_keywords[np.argmax(cosine_scores)]

# input = "Sorry, I need to go now."
# print(check_similarity(input, exit_embeddings))
