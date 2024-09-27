import streamlit as st

def side_panel():
    """
    Creates a side bar, take API keys as input and return the keys
    """
    with st.sidebar:
        st.markdown(
                "## How to use\n"
                "1. Enter your OpenAI & Llama Cloud API keys below🔑\n" 
                "2. Upload a pdf, docx, or txt file📄\n"
                "3. Ask a question about the document💬\n"
            )

        openai_api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Paste your OpenAI API key here (sk-...)",
                help="You can get your API key from https://platform.openai.com/account/api-keys.",
                value=st.session_state.get("OPENAI_API_KEY", "") # os.environ.get("OPENAI_API_KEY", None)
            )
        st.session_state["OPENAI_API_KEY"] = openai_api_key_input
        llama_api_key_input = st.text_input(
                "Llama Cloud API Key",
                type="password",
                placeholder="Paste your Llama Cloud API key here (lxx-...)",
                help="You can get your API key from https://cloud.llamaindex.ai/api-key.",
                value=st.session_state.get("LLAMA_API_KEY", "")
            )
        st.session_state["LLAMA_API_KEY"] = llama_api_key_input

        # st.button("Submit")

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

        return [openai_api_key_input, llama_api_key_input]