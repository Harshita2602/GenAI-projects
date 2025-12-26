import streamlit as st
import os
import time

# Langchain/community imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Environment ad configuration
from dotenv import load_dotenv
load_dotenv()  

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

if "vectors" not in st.session_state:
    
    # create local embeddings via Ollama
    st.session_state.embeddings=OllamaEmbeddings() 
    
    # load docs from a website
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/") 
    st.session_state.docs=st.session_state.loader.load()
    
    #split documents into chunks for RAG (preserved context via overlap)
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    
    #used only first 50 docs to keep dmo fast, adjust as needed
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    
    #build faiss vector index from chunked docs plus embeddings
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("ChatGroq Demo")

#initialize GROQ LLM (Mixtral)
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")

#prompt template: instruct model to answer only from context
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

#chain that stuff retrieved docs into the prompt and calls the llm
document_chain = create_stuff_documents_chain(llm, prompt)

#build retriever from FAISS store
retriever = st.session_state.vectors.as_retriever()

#combine retriever plus document chain into single retriever chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start) #log response time in console
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
    