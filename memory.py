from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
embedding = HuggingFaceEndpointEmbeddings(
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


# Only create FAISS if it does not exist
if not os.path.exists(DB_FAISS_PATH):

    docs = load_pdf_files("files")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70
    )

    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )

    db.save_local(DB_FAISS_PATH)

    print("FAISS index created successfully")

else:
    print("FAISS index already exists")