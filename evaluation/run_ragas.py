import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (context_precision,context_recall,faithfulness,       # NEW
    answer_relevancy)
from ragas.run_config import RunConfig

# Create a "Slow and Steady" configuration
run_config = RunConfig(
    timeout=180,           # Increase timeout to 3 minutes per question
    max_retries=10,        # Give it more chances to get the JSON right
    max_workers=1          # CRITICAL: Evaluate 1 question at a time for local models
)

# 1. Import the wrappers needed for local models
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import your existing pipeline objects
from app.rag_pipeline import run_rag, llm, embeddings 

# 2. Wrap your local Ollama and HuggingFace models
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Load evaluation data
with open("evaluation/eval_data.json") as f:
    eval_data = json.load(f)

records = []
for item in eval_data:
    # This calls your existing RAG pipeline
    result = run_rag(item["question"])
    
    records.append({
        "question": item["question"],
        "answer": result["answer"],
        "contexts": result["contexts"],
        "ground_truth": item["ground_truth"]
    })

dataset = Dataset.from_list(records)

# 3. Pass the wrapped models to the evaluate function
# This prevents Ragas from looking for an OpenAI API Key
results = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,       # Checks for hallucinations
        answer_relevancy   # Checks if answer stays on topic
    ],
    llm=ragas_llm,           # Use your local Llama 3.1 as the 'judge'
    embeddings=ragas_embeddings, # Use your local embeddings for vector math
    run_config=run_config
)

print("\n--- RAGAS EVALUATION RESULTS ---")
print(results)
