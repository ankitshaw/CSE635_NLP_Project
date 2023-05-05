import os
from haystack.document_stores import ElasticsearchDocumentStore

# Get the host where Elasticsearch is running, default to localhost
# host = os.environ.get("ELASTICSEARCH_HOST", "localhost")


def get_document_store():
    document_store = ElasticsearchDocumentStore(
        host="my-deployment-9c9938.es.us-central1.gcp.cloud.es.io",
        port="9243",
        scheme="https",
        username="elastic",
        password="7UpMOQEaTEAChvhuqlHhonNR",
        index="document2",
        embedding_dim=768
    )
    return document_store

def get_retriever(document_store,type):
    from haystack.nodes import DensePassageRetriever, EmbeddingRetriever

    if type == "dpr":
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
            passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
        )
        return retriever
    else:
        retriever = EmbeddingRetriever(
            document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers"
        )
        # document_store.update_embeddings(retriever)
    
        return retriever