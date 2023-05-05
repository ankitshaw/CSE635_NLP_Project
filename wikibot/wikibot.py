from util import get_document_store, get_retriever
from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline

document_store = get_document_store()
retriever = get_retriever(document_store=document_store, type="embed")
generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
pipe = GenerativeQAPipeline(generator, retriever)


def get_response(query):
    result = pipe.run(
        query=query, params={"Retriever": {"top_k": 3}}
    )
    print(result)
    return result['answers'][0].answer

# print(get_response("I will be visiting India in December?"))