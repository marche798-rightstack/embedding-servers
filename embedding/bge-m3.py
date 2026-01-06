from typing import List, Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import torch

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load BGE-M3 embedding model once at startup
print("Loading BGE-M3 embedding model")
model = SentenceTransformer("BAAI/bge-m3", device=device)
print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")


class EmbeddingRequest(BaseModel):
    # OpenAI-compatible: accept string, list of strings, token list, or list of token lists
    input: Union[str, List[str], List[int], List[List[int]]]
    model: Optional[str] = None
    dimensions: Optional[int] = None


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    raw = request.input

    # Normalize input to a list of strings
    if isinstance(raw, str):
        texts = [raw]
    elif isinstance(raw, list):
        if raw and all(isinstance(x, int) for x in raw):
            # Token list -> join as space-separated string (adjust as needed)
            texts = [" ".join(map(str, raw))]
        elif raw and all(isinstance(x, list) for x in raw):
            # List of token lists -> convert each to a string
            texts = [" ".join(map(str, seq)) for seq in raw]
        else:
            # Assume list of strings; coerce any non-str to str
            texts = [str(t) for t in raw]
    else:
        texts = [str(raw)]

    embeddings = model.encode(texts, batch_size=1, normalize_embeddings=True)

    data = [
        {"object": "embedding", "embedding": emb.tolist(), "index": i}
        for i, emb in enumerate(embeddings)
    ]

    usage_tokens = sum(len(str(t).split()) for t in texts)
    return {
        "object": "list",
        "data": data,
        "model": request.model or "bge-m3",
        "usage": {"prompt_tokens": usage_tokens, "total_tokens": usage_tokens},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)

