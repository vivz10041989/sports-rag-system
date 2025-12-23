import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    text = text.lower()                       # normalize case
    text = re.sub(r"\s+", " ", text)           # remove extra spaces
    text = re.sub(r"[^a-z0-9.,:/ ]", "", text)  # remove noise
    return text.strip()

for file in RAW_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_text(raw_text)

    output_file = PROCESSED_DIR / file.name
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Processed: {file.name}")


DATA_DIR = Path("data/processed")

# 1. Load documents with metadata
documents = []

for file in DATA_DIR.glob("*.txt"):
    loader = TextLoader(str(file), encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata = {
            "source": file.name,
            "type": "sports_document"
        }
        documents.append(doc)

print(f"Loaded {len(documents)} documents")

# 2. Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Save index locally
vectorstore.save_local("faiss_index")

print("FAISS index created and saved")

