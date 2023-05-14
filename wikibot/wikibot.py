from .util import get_document_store, get_retriever
from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
import os
from haystack.document_stores import ElasticsearchDocumentStore

# Get the host where Elasticsearch is running, default to localhost
# host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = get_document_store()
retriever = get_retriever(document_store=document_store, type="embed")
generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
pipe = GenerativeQAPipeline(generator, retriever)


def get_wiki_response(query):
    result = pipe.run(
        query=query, params={"Retriever": {"top_k": 3}}
    )
    print(result)
    return result['answers'][0].answer

def get_wiki_batch_response(queries):
    result = pipe.run_batch(
        queries=queries, params={"Retriever": {"top_k": 3}}
    )
    # print(result['answers'])
    ans = []
    for res in result['answers']:
        # print(res)
        ans.append(res[0].answer)
    
    return ans
    # return result['answers'][0].answer
# print(get_response("I will be visiting India in December?"))
# print(get_wiki_batch_response(["who is babur?","what is answer to life"]))
