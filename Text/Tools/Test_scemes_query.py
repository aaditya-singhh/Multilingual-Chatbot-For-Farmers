# Tools/test_schemes_query.py
from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

BASE = Path(__file__).resolve().parents[1]
ENV = BASE / ".env"
if ENV.exists():
    load_dotenv(ENV)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set")

PERSIST_DIR = BASE / "Indexes" / "schemes_chroma"

print("Loading Chroma collection...")
emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
vs = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=emb,
    collection_name="schemes"
)

retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def ask(q: str):
    docs = retriever.get_relevant_documents(q)
    print(f"\nQ: {q}\nTop {len(docs)} chunks:\n" + "-"*40)
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content[:700]}{'...' if len(d.page_content) > 700 else ''}\n")

if __name__ == "__main__":
    ask("What is PM-KISAN? Eligibility & benefit amount?")
    ask("How does PMFBY (crop insurance) claim process work after flood damage?")
