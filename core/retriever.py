from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.postprocessor import SentenceTransformerRerank

def create_query_engine(index):
    DEFAULT_TEXT_QA_PROMPT_TMPL = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
        DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    )

    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
    # reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-large") node_postprocessors=[reranker]
    query_engine = index.as_query_engine(similarity_top_k=3, verbose=True, 
                                         node_postprocessors=[reranker], 
                                         text_qa_template=DEFAULT_TEXT_QA_PROMPT)
    return query_engine