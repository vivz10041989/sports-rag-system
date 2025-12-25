##from fastapi import FastAPI

##app = FastAPI()

####@app.get("/")
##def health_check():
##    return {"status": "RAG system running"}

from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import rag_answer

app = FastAPI(title="Sports RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    answer = rag_answer(request.question)
     # 🔹 NORMALIZE ANSWER (API layer)
    answer = " ".join(answer.split())
    return QueryResponse(answer=answer)

@app.get("/")
def health_check():
    return {"status": "Sports RAG API running"}
