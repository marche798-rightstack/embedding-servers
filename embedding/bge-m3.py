from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from FlagEmbedding import BGEM3FlagModel
import uvicorn
import torch
import base64
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BGE-M3 Embedding Server")

# CORS 설정 (필요시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# 지원 모델 매핑
MODEL_MAP = {
    "bge-m3": "BAAI/bge-m3",
}

# BGE-M3 임베딩 차원
BGE_M3_DIMENSION = 1024

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str = Field(default="bge-m3")
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")
    dimensions: Optional[int] = Field(default=None, gt=0, le=BGE_M3_DIMENSION)
    user: Optional[str] = None

model_cache: dict[str, BGEM3FlagModel] = {}

def get_model(model_name: str) -> BGEM3FlagModel:
    if model_name not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"unsupported model: {model_name}")
    if model_name not in model_cache:
        logger.info(f"Loading model: {MODEL_MAP[model_name]}")
        try:
            model_cache[model_name] = BGEM3FlagModel(
                MODEL_MAP[model_name],
                use_fp16=True,  # GPU 최적화
                device=device
            )
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
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

        # 문자열 배열
        return [str(t).strip() for t in raw if str(t).strip()]

    # 기타는 문자열 캐스팅
    return [str(raw)]

def trim_dimensions(vec: np.ndarray, dim: Optional[int]) -> np.ndarray:
    if dim is None:
        return vec
    return vec[:, :dim]

def encode_output(vectors: np.ndarray, encoding_format: str):
    """각 임베딩을 개별적으로 인코딩"""
    if encoding_format == "float":
        return vectors.tolist()
    
    # base64: 각 행을 개별적으로 인코딩
    encoded_list = []
    for vec in vectors:
        as_bytes = vec.astype(np.float32).tobytes()
        encoded_list.append(base64.b64encode(as_bytes).decode("ascii"))
    return encoded_list

@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    try:
        texts = normalize_input(req.input)

        # 간단한 길이 제한
        if len(texts) > 2048:
            raise HTTPException(status_code=400, detail="too many inputs (max 2048)")

        model = get_model(req.model)

        # FlagEmbedding 방식으로 임베딩 생성
        # BGE-M3 최적 batch_size는 12
        result = model.encode(
            texts,
            batch_size=min(12, len(texts)),  # BGE-M3 최적값
            max_length=8192,
            return_dense=True,      # Dense 임베딩만
            return_sparse=False,    # Sparse는 OpenAI API에서 지원 안 함
            return_colbert_vecs=False  # ColBERT도 제외
        )
        
        # dense_vecs 추출
        embeddings = result['dense_vecs']
        
        # dimensions 처리
        if req.dimensions:
            embeddings = embeddings[:, :req.dimensions]

        # 정규화 (OpenAI처럼)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Zero division 방지
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        # encoding_format 처리
        encoded = encode_output(embeddings, req.encoding_format)

        # 응답 생성
        data = []
        if req.encoding_format == "float":
            for i, emb in enumerate(encoded):
                data.append({
                    "object": "embedding",
                    "embedding": emb,
                    "index": i
                })
        else:
            # base64: 각 임베딩마다 별도의 base64 문자열
            for i, emb_b64 in enumerate(encoded):
                data.append({
                    "object": "embedding",
                    "embedding": emb_b64,
                    "index": i
                })

        # usage 추정
        usage_tokens = sum(len(t.split()) for t in texts)

        return {
            "object": "list",
            "data": data,
            "model": req.model,
            "usage": {
                "prompt_tokens": usage_tokens,
                "total_tokens": usage_tokens
            }
        }
    
    except HTTPException:
        raise  # HTTPException은 그대로 전달
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 워밍업"""
    logger.info("Starting BGE-M3 Embedding Server...")
    try:
        # 모델 미리 로드 (첫 요청 지연 방지)
        model = get_model("bge-m3")
        # 워밍업 임베딩
        _ = model.encode(
            ["warmup"],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        logger.info("Model warmup completed")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")

@app.get("/v1/models")
async def list_models():
    """OpenAI 호환"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1234567890,
                "owned_by": "BAAI"
            }
            for model_id in MODEL_MAP.keys()
        ]
    }

@app.get("/health")
async def health():
    """헬스체크 (상세 정보 포함)"""
    health_info = {
        "status": "healthy",
        "device": device,
        "models": list(MODEL_MAP.keys()),
        "models_loaded": list(model_cache.keys()),
    }
    
    # GPU 정보 추가
    if torch.cuda.is_available():
        health_info["gpu"] = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        }
    else:
        health_info["gpu"] = {"available": False}
    
    return health_info

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BGE-M3 Embedding Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )