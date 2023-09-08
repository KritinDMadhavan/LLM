# main.py
import os
from tables import tables
from chatmodel import extract_entities
from langchain.embeddings import OpenAIEmbeddings

from docstore_module import document_splitter
from langchain.vectorstores import Chroma

# Split documents
docs = document_splitter(tables)

openai_api_key = ''
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Set up Chroma database
db = Chroma.from_documents(
    docs, embedding=embeddings,
    persist_directory="./chroma_db"
)

# Extract entities using chat model
ques = extract_entities(db, "Who are the top articles this month and find the author who has the maximumm top articles",docs)
print(ques)
