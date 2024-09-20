import streamlit as st
import nest_asyncio
import os
import io
# import pdfplumber
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_parse import LlamaParse
from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
# from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Load environment variables
load_dotenv()

# Extract API keys from environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLAMA_CLOUD_API_KEY =  os.environ["LLAMA_CLOUD_API_KEY"]
AZ_BLOB_KEY = os.environ["AZ_BLOB_KEY"]
AZ_BLOB_ACC_NAME = os.environ["AZ_BLOB_ACC_NAME"]
AZ_CONTAINER_NAME = "uploads"

# Create nodes/chunks for each page in the PDF
def get_page_nodes(docs, separator="\n---\n"):
    """Split each document into page node, by separator."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)
    return nodes

# def parse_pdf_with_pdfplumber(file, table=False):
#     with pdfplumber.open(file) as pdf:
#         pdf_text = ""
#         for page in pdf.pages:
#             pdf_text += page.extract_text()       
#         if table:
#             pdf_table = []
#             for page in pdf.pages:
#                 pdf_table.append(page.extract_table())
#         else:
#             pass
#     return pdf_text

# def langchain_transform(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

def llama_index_transform(text):
    doc = Document(text=text)
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents([doc], show_progress=False)
    return nodes

def parse_with_llama_parser(file_path):
    document = LlamaParse(result_type="markdown").load_data(file_path=file_path)
    return document
    
def transform(document, llm, embed_model):
    page_nodes = get_page_nodes(document)
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(document)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    return base_nodes, objects, page_nodes

def vector_store(nodes):
    recursive_index = VectorStoreIndex(nodes=nodes)
    return recursive_index

def retriever(index):
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
    # reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-large") node_postprocessors=[reranker]
    query_engine = index.as_query_engine(similarity_top_k=3, verbose=True, 
                                             text_qa_template=DEFAULT_TEXT_QA_PROMPT)
    return query_engine


def main():
    # setup global variables
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o-mini-2024-07-18")
    Settings.llm = llm
    Settings.embed_model = embed_model

    # app interface
    st.header("📝 RAG based Chat with PDF")
    st.info("Please upload a file first")
    uploaded_file = st.file_uploader("Upload a file", type="pdf")
    if uploaded_file is not None:
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Upload the file to the Azure Blog Storage (only if the file is not present there already)
                with open(uploaded_file.name, "wb") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                blob_service_client = BlobServiceClient(
                    f"https://{AZ_BLOB_ACC_NAME}.blob.core.windows.net", credential=AZ_BLOB_KEY
                )
                blob_client = blob_service_client.get_blob_client(container=AZ_CONTAINER_NAME, blob=uploaded_file.name)
                try:
                    blob_properties = blob_client.get_blob_properties()
                    file_exists = True
                except Exception as e:
                    file_exists = False
                if file_exists:
                    print("File found in Azure Blob Storage. Downloading...")
                else:
                    print("File not found in Azure Blob Storage. Uploading...")
                    with open(uploaded_file.name, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                        st.success("File uploaded successfully!")
                # Download the file from Azure Blob Storage
                local_file_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(local_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                    st.success("File downloaded successfully!")

                
                # st.write(f"File is accessible at: {file_url}")
                st.session_state.markdown_document = parse_with_llama_parser(local_file_path)
                st.session_state.base_nodes, st.session_state.objects, st.session_state.page_nodes = transform(st.session_state.markdown_document, llm, embed_model)
                # st.session_state.raw_text = parse_pdf_with_pdfplumber(uploaded_file)
                # st.session_state.nodes = llama_index_transform(st.session_state.raw_text)
                st.session_state.index = vector_store(nodes=st.session_state.base_nodes + st.session_state.objects + st.session_state.page_nodes)
                st.session_state.retriever = retriever(st.session_state.index)
                st.success("Done! Now ask me a question")

    st.caption("User:")
    user_question = st.text_input(
        "Ask something from the file",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )
    
    if user_question:
        with st.spinner("Responding..."):
            st.caption("Chat Agent:")
            st.session_state.response = st.session_state.retriever.query(user_question)
            st.info(st.session_state.response)

if __name__ == '__main__':
    main()

