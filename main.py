#!/usr/bin/env python3
from langchain.chains import RetrievalQA
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time

model = os.environ.get("MODEL", "mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

def main():
    st.title("Language Model Chat")

    # Sidebar options
    st.sidebar.title("Settings")
    hide_source = st.sidebar.checkbox("Hide Source Documents")
    mute_stream = st.sidebar.checkbox("Mute Stream")

    # Parse the command line arguments
    args = parse_arguments(hide_source, mute_stream)

    st.sidebar.markdown("---")

    embeddings_model_name = "all-MiniLM-L6-v2"  # Modify as needed
    persist_directory = "db"  # Modify as needed
    target_source_chunks = 4  # Modify as needed

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = Ollama(model="mistral")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    query = st.text_input("Enter a query:")
    if query:
        res = qa(query)
        answer = res['result']
        docs = res.get('source_documents', [])

        # Display the result
        st.write("\n\n> **Question:**")
        st.write(query)
        st.write("\n\n> **Answer:**")
        st.write(answer)

        if docs:
            top_document = docs[0]  # Get the top document
            st.write("\n\n> **Top Source Document:**")
            st.write(f"\n> {top_document.metadata.get('source', 'Unknown')}:")
            st.write(top_document.page_content)

def parse_arguments(hide_source, mute_stream):
    class Args:
        def __init__(self):
            self.hide_source = hide_source
            self.mute_stream = mute_stream

    return Args()


if __name__ == "__main__":
    main()
