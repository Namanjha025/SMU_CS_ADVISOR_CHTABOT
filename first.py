import os
os.environ["google_api_key"] = "AIzaSyCsgnZBjnr2obV5AtyN_M_rouHW6cjL7Xc"

from langchain.llms import GooglePalm
llm = GooglePalm( temperature=0)


#load document
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("C:/Users/naman/Desktop/Langchain/SMU_CHAT_APP/top1/Program_ Computer Science, M.S. - Southern Methodist University - Acalog ACMS™.pdf")
#to load the document do this
document = loader.load()


#split the document
from langchain.text_splitter import RecursiveCharacterTextSplitter
chunk_size = 500
chunk_overlap = 20

r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
split = r_splitter.split_documents(document)

#once the document is splitted into chunks, let us now create embeddings of these chunks
#we will use googlepaLm embedding models to create embeddings of split
from langchain.embeddings import GooglePalmEmbeddings
embeddings = GooglePalmEmbeddings()


#let us create vector db to store these embeddings. This vectordb is stored in memory. 

from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQA

vectordb = FAISS.from_documents(documents=split, embedding=embeddings)
#launching the QA chain
qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore = vectordb)



import streamlit as st
st.title("MS in CS SMU QA")


question = st.text_input("Question: ")

if question:
    st.header("Answer: ")
    st.write((qa.run(question)))
   



    







