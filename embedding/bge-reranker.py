from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import CrossEncoder
import uvicorn
import torch

# Limit CPU threads
#torch.set_num_threads(9)
print(f"PyTorch threads set to: {torch.get_num_threads()}")

app = FastAPI()

#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load proper sequence classification model
# print("Loading Qwen3-Reranker-0.6B-seq-cls on CPU...")
# model = CrossEncoder(
#     'tomaarsen/Qwen3-Reranker-0.6B-seq-cls',
#     max_length=8192,
#     device=device
# )
model = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device)
print("Model loaded!")

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

@app.post("/v1/rerank")
async def rerank(request: RerankRequest):
    results = []

    # Process one by one with explicit batch_size=1
    for i, doc in enumerate(request.documents):
        score = model.predict([[request.query, doc]], batch_size=10)[0]
        results.append({"index": i, "relevance_score": float(score)})

    # Sort by score descending
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)

