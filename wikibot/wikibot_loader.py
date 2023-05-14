from .util import get_document_store
import pandas as pd
from haystack import Document
import ast

document_store = get_document_store()

df = pd.read_csv("simplewiki.csv")
# Minimal cleaning
df.fillna(value="", inplace=True)
df = df.rename(columns={'paragraphs':'text'})
print(df.head())



# Use data to initialize Document objects
titles = list(df["title"].values)
texts = list(df["text"].values)


documents = []
for title, text in zip(titles, texts):
    text = ast.literal_eval(text)
    for txt in text:
      documents.append(Document(content=txt, meta={"name": title or ""}))

# Delete existing documents in documents store
document_store.delete_documents()

# # Write documents to document store
document_store.write_documents(documents)


from haystack.nodes import DensePassageRetriever, EmbeddingRetriever

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
    passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
)

document_store.update_embeddings(retriever)

# retriever = EmbeddingRetriever(
#     document_store=document_store,
#    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
#    model_format="sentence_transformers"
# )
# document_store.update_embeddings(retriever)