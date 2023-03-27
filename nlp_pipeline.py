from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.pipeline.entity_linker import DEFAULT_NEL_MODEL
from spacy.kb import KnowledgeBase
from spacy.kb import InMemoryLookupKB
import spacy
from wikidata.client import Client
import wikipedia

client = Client()
# url = entity.data['sitelinks']['frwiki']['url']

# print(url)

config_ner = {
   "moves": None,
   "update_with_oracle_cut_size": 100,
   "model": DEFAULT_NER_MODEL,
   "incorrect_spans_key": "incorrect_spans",
}

config_nel = {
   "labels_discard": [],
   "n_sents": 0,
   "incl_prior": True,
   "incl_context": True,
   "model": DEFAULT_NEL_MODEL,
   "entity_vector_length": 64,
   "get_candidates": {'@misc': 'spacy.CandidateGenerator.v1'},
   "threshold": None,
}

nlp = spacy.load('en_core_web_lg')

nlp.add_pipe("entityLinker", last=True)

def get_nel(prompt):
    doc = nlp(prompt)
    # print(extracted_ner)
    all_linked_entities = doc._.linkedEntities
    # print(all_linked_entities)
    links= []
    for entity in all_linked_entities:
        # print(entity.get_url())
        entity_wiki = client.get(entity.get_url().split("/")[-1], load=True)
        url = entity_wiki.data['sitelinks']['enwiki']['url']
        links.append(url)
    return links #returns list of wikipedis links for entity


