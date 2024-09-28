import os
from dotenv import load_dotenv
import openai
import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from core.doc_processor import DocumentProcessor
from core.retriever import create_query_engine
from core.index import vector_store

# -------------- App interface - Side bar --------------
with st.sidebar:
    st.markdown(
            "## How to use\n"
            "1. Enter your OpenAI key below🔑\n" 
            "2. Upload a pdf, docx, or txt file📄\n"
            "3. Ask a question about the document💬\n"
        )

    openai.api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
            value=st.session_state.get("OPENAI_API_KEY", "") # os.environ.get("OPENAI_API_KEY", None)
        )
    if not openai.api_key.startswith('sk-'):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Proceed to upload your file!', icon='👉')

    st.markdown("---")
    st.markdown("# About")
    st.markdown(
        "📝 DocuAssist AI assist you in going through large and complex documentation by answering your \
        questions related to the document."
    )
    st.markdown(
        "This tool is a work in progress. "
        "Feedback and suggestions are most welcome!"
    )
    st.markdown("Made by [suvkp](https://github.com/suvkp)")
    st.markdown("---")
# ---------------------------------------------------------------
# setup global variables
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm
Settings.embed_model = embed_model

# -------------- App interface - header & uploader --------------
st.header("📝 DocuAssist AI")

uploaded_file = st.file_uploader("Upload a file", type=["pdf","xlsx","doc"], disabled=(len(openai.api_key)==0))

# document_processed = False
if uploaded_file is not None:
    if st.button("Submit & Process"):
        with st.spinner("Reading ..."):
            # download the file to a local folder to pass it LlamaParse
            file_path = os.path.join('resource/', uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.base_nodes, st.session_state.objects, st.session_state.page_nodes = DocumentProcessor(llama_parser_api_key=st.secrets["LLAMA_CLOUD_API_KEY"]).transform(file_path, llm, embed_model)
            st.session_state.index = vector_store(nodes=st.session_state.base_nodes + st.session_state.objects + st.session_state.page_nodes)
            st.session_state.retriever = create_query_engine(st.session_state.index)
            st.success("Done! Now ask me a question.")
            # document_processed = True

# ---------------- App interface - Chat function -----------------
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display initial massage of the system right after user uploads the document
# if document_processed:
#     with st.chat_message("assistant"):
#         init_sys_prompt = "Give a short summary and outline the content of the uploaded document. \
#         Respond to this by saying - Thanks for uploading the document. Here's the short summary and content \
#         of the document ..."
#         st.session_state.init_sys_response = st.session_state.retriever.query(init_sys_prompt)
#         st.markdown(st.session_state.init_sys_response)
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": st.session_state.init_sys_response})

# React to user input
if prompt := st.chat_input("Ask me a question", disabled= not uploaded_file):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.retriever.query(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # st.write(type(response.response))
        response_formatted = response.response.replace("$", "\$") # remove any $ to prevent latex detection in markdown
        st.markdown(response_formatted)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_formatted})
