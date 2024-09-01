import getpass
import os
from langchain_groq import ChatGroq
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.llms.openai import OpenAI

def check_env_var(variable_name):
    if variable_name in os.environ:
        return True
    else:
        return False
    

def LLMCallFunc(llm):
    # Split the source text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(source_text)
    docs = [Document(page_content=t) for t in texts[:3]]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(docs)
    return summary["output_text"]

st.title('Text Summerizer using LangChain & Groq')
# del os.environ['GROQ_API_KEY']
# if not check_env_var('GROQ_API_KEY'):
#     os.environ["GROQ_API_KEY"] = getpass.getpass()
#     llm_api = st.text_input("Groq or other LLM API Key", type="password")
option = st.selectbox(
    "Select the API provider ",
    ("Selection", "Groq", "OpenAI")
)
llm_api = st.text_input("Groq or other LLM API Key", type="password")

source_text = st.text_area("Source Text", height=200)
# Check if the 'Summarize' button is clicked
if st.button("Summarize"):
    if not llm_api.strip() or not source_text.strip():
        st.write(f"Please complete the missing fields.")
    else:
        try:
            
            if option == "Groq":
                llm = ChatGroq(model="llama3-8b-8192", api_key=llm_api)
                summary=LLMCallFunc(llm)
            elif option == "OpenAI":
                llm = OpenAI(temperature=0, openai_api_key=llm_api)
                summary=LLMCallFunc(llm)
            else:
                summary="Please select the API provider"
            # Display summary
            st.write(summary)
        except Exception as e:

            st.write(f"Oops, Something just broke :(")
            st.write(f"{e}")