from bs4 import BeautifulSoup
import requests
import nltk
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from time import sleep
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class TopicBot():

    # initialize bot
    def __init__(self):
        self.end_chat = False
        self.got_topic = False
        self.do_not_respond = True

        self.title = None
        self.topics = {}
        self.text_data = []
        self.sentences = []
        self.para_indices = []
        self.current_sent_idx = None
        self.topical_data = {}

        self.punctuation_dict = str.maketrans({p: None for p in punctuation})
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stopwords = nltk.corpus.stopwords.words('english')

    def generator(self, prompt, links):
        for link in links:
            if link not in self.topical_data:
                self.scrape_wiki(link)
            self.topical_data[link].append(prompt)

        print("get responses")
        responses = self.responses(links)
        print(responses)
        responses.append(prompt)
        return self.ranker(responses)

    def ranker(self, responses):
        if len(responses) > 1:
            vectorizer = TfidfVectorizer(tokenizer=self.preprocess)
            tfidf = vectorizer.fit_transform(responses)
            scores = cosine_similarity(tfidf[-1], tfidf)
            current_sent_idx = scores.argsort()[0][-2]
            scores = scores.flatten()
            scores.sort()
            value = scores[-2]
            if value != 0:
                return responses[current_sent_idx]
            else:
                return ""
        else:
            return ""

    # respond method - to be called internally

    def responses(self, links):
        # tf-idf-modeling
        soi = []
        for link in links:
            vectorizer = TfidfVectorizer(tokenizer=self.preprocess)
            tfidf = vectorizer.fit_transform(self.topical_data[link])
            scores = cosine_similarity(tfidf[-1], tfidf)
            if len(scores) > 2:
                current_sent_idx = scores.argsort()[0][-2]
            else:
                current_sent_idx = 0
                print("scores:", scores)
            scores = scores.flatten()
            scores.sort()
            if(len(scores)<2):
                value = 0
            else:
                value = scores[-2]
            if value != 0:
                soi.append(self.topical_data[link][current_sent_idx])
            del self.topical_data[link][-1]
        return soi

    def scrape_wiki(self, link):
        try:
            data = requests.get(link).content
            soup = BeautifulSoup(data, 'html.parser')
            p_data = soup.findAll('p')
            dd_data = soup.findAll('dd')
            p_list = [p for p in p_data]
            dd_list = [dd for dd in dd_data]
            sentences = []
            text_data = []
            for tag in p_list+dd_list:  # +li_list:
                a = []
                for i in tag.contents:
                    if i.name != 'sup' and i.string != None:
                        stripped = ' '.join(i.string.strip().split())
                        a.append(stripped)
                text_data.append(' '.join(a))

            for i, para in enumerate(text_data):
                sentences = nltk.sent_tokenize(para)
                sentences.extend(sentences)
                index = [i]*len(sentences)
                self.para_indices.extend(index)

            self.topical_data[link] = sentences

            self.topics[soup.find('h1').string] = link
            self.got_topic = True
        except Exception as e:
            print('ChatBot >>  Error: {}. \
            Please input some other topic!'.format(e))

    def preprocess(self, text):
        # remove punctuations
        text = text.lower().strip().translate(self.punctuation_dict)
        # tokenize into words
        words = nltk.word_tokenize(text)
        # remove stopwords
        words = [w for w in words if w not in self.stopwords]
        # lemmatize
        return [self.lemmatizer.lemmatize(w) for w in words]


tc = TopicBot()
