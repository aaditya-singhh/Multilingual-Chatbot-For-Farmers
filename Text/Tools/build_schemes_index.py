# Tools/build_schemes_index.py
from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

BASE = Path(__file__).resolve().parents[1]  # -> D:\Chatbot
ENV = BASE / ".env"
if ENV.exists():
    load_dotenv(ENV)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment or .env")

PDF_PATH = BASE / "Data" / "Scemes for welfare of farmers.pdf"   # <— your PDF
if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

PERSIST_DIR = BASE / "Indexes" / "schemes_chroma"
PERSIST_DIR.parent.mkdir(parents=True, exist_ok=True)

print("Loading PDF...")
docs = PyPDFLoader(str(PDF_PATH)).load()

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

print("Embedding with Google (text-embedding-004)...")
emb = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

print("Writing to Chroma (disk-persistent)...")
_ = Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    persist_directory=str(PERSIST_DIR),
    collection_name="schemes"
)

print(f"Persisted vector DB to: {PERSIST_DIR}")
print("Done.")
