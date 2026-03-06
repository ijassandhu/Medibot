from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
embedding_model = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token = token,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

DB_FAISS_PATH = 'vectorstore/db/faiss'

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

try:
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("FAISS loaded successfully")

except Exception as e:
    print("FAISS load failed, rebuilding index:", e)

    docs = load_pdf_files("files")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70
    )

    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    db.save_local(DB_FAISS_PATH)

    print("FAISS rebuilt successfully")