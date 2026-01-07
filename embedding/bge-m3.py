from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import uvicorn
import torch
import base64
import numpy as np

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 지원 모델 매핑
MODEL_MAP = {
  "bge-m3": "BAAI/bge-m3",
  # 필요시 추가: "another-id": "hf-repo-or-path"
}

class EmbeddingRequest(BaseModel):
  input: Union[str, List[str], List[int], List[List[int]]]
  model: str = Field(default="bge-m3")
  encoding_format: str = Field(default="float", pattern="^(float|base64)$")
  dimensions: Optional[int] = Field(default=None, gt=0)
  user: Optional[str] = None  # 전달만 함, 사용은 안 함

model_cache: dict[str, SentenceTransformer] = {}

def get_model(model_name: str) -> SentenceTransformer:
  if model_name not in MODEL_MAP:
      raise HTTPException(status_code=400, detail=f"unsupported model: {model_name}")
  if model_name not in model_cache:
      model_cache[model_name] = SentenceTransformer(MODEL_MAP[model_name], device=device)
  return model_cache[model_name]

def normalize_input(raw) -> List[str]:
  # 문자열 한 건
  if isinstance(raw, str):
      if not raw.strip():
          raise HTTPException(status_code=400, detail="input cannot be empty")
      return [raw]

  # 리스트
  if isinstance(raw, list):
      if not raw:
          raise HTTPException(status_code=400, detail="input list cannot be empty")

      # 토큰 배열 -> 공백 구분 문자열
      if all(isinstance(x, int) for x in raw):
          return [" ".join(map(str, raw))]

      # 토큰 배열의 배열
      if all(isinstance(x, list) and x and all(isinstance(t, int) for t in x) for x in raw):
          return [" ".join(map(str, seq)) for seq in raw]

      # 문자열 배열 (나머지는 str로 캐스팅)
      return [str(t) for t in raw]

  # 기타는 문자열 캐스팅
  return [str(raw)]

def trim_dimensions(vec: np.ndarray, dim: Optional[int]) -> np.ndarray:
  if dim is None:
      return vec
  return vec[:, :dim]

def encode_output(vectors: np.ndarray, encoding_format: str):
  if encoding_format == "float":
      return vectors.tolist()
  # base64 로우 인코딩: float32 -> bytes -> base64
  as_bytes = vectors.astype(np.float32).tobytes()
  return base64.b64encode(as_bytes).decode("ascii")

@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
  texts = normalize_input(req.input)

  # 간단한 길이 제한 (스펙: per-input 8192 tokens, total 300k)
  if len(texts) > 2048:
      raise HTTPException(status_code=400, detail="too many inputs (max 2048)")

  model = get_model(req.model)

  # 배치 크기 적당히 설정
  embeddings = model.encode(
      texts,
      batch_size=min(max(8, len(texts)), 64),
      normalize_embeddings=True,
  )

  # dimensions 처리
  embeddings = trim_dimensions(np.array(embeddings), req.dimensions)

  # 응답 encoding_format 처리
  encoded = encode_output(embeddings, req.encoding_format)

  data = []
  if req.encoding_format == "float":
      for i, emb in enumerate(encoded):
          data.append({"object": "embedding", "embedding": emb, "index": i})
  else:
      # base64일 때는 하나의 base64 문자열만 반환 (OpenAI 스펙: embedding이 base64 문자열)
      for i in range(len(texts)):
          data.append({"object": "embedding", "embedding": encoded, "index": i})

  # usage는 정확 토큰 카운트가 어려워 단순 추정치
  usage_tokens = sum(len(t.split()) for t in texts)

  return {
      "object": "list",
      "data": data,
      "model": req.model,
      "usage": {"prompt_tokens": usage_tokens, "total_tokens": usage_tokens},
  }

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8010)

