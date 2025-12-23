from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.1",
    temperature=0
)

# Load embeddings model (must match ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


def retrieve_context(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    return context

def build_prompt(context: str, question: str) -> str:
    prompt = f"""
You are a sports analytics assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt

def generate_answer(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response


def rag_answer(question: str) -> str:
    context = retrieve_context(question)
    prompt = build_prompt(context, question)
    return generate_answer(prompt)



