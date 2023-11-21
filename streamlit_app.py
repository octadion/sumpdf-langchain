from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
# llm = OpenAI(temperature=0)
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=True)
        summary = chain.run(texts)
        summaries.append(summary)
        os.remove(temp_path)
    return summaries

st.set_page_config("SumPDF-LangChain", page_icon=":books:")
st.title("PDF Summarizer")

pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if pdf_files:
    if st.button("Generate"):
        st.write("Summaries:")
        summaries = summarize_pdfs_from_folder(pdf_files)
        for i, summary in enumerate(summaries):
            st.write(f"Summary for PDF {i+1}:")
            st.write(summary)

