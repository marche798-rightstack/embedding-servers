from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from FlagEmbedding import FlagReranker
import uvicorn
import torch
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BGE Reranker Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# 지원 모델 매핑
MODEL_MAP = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}

class Document(BaseModel):
    """재순위화할 문서"""
    text: str
    index: Optional[int] = None  # 원본 인덱스 (선택)

class RerankRequest(BaseModel):
    """재순위화 요청"""
    query: str = Field(..., description="검색 쿼리")
    documents: List[str] = Field(..., description="재순위화할 문서 리스트")
    model: str = Field(default="bge-reranker-v2-m3")
    top_n: Optional[int] = Field(default=None, description="상위 N개만 반환")

class RerankResult(BaseModel):
    """재순위화 결과"""
    index: int
    relevance_score: float
    document: str

class RerankResponse(BaseModel):
    """재순위화 응답"""
    model: str
    results: List[RerankResult]
    usage: dict

model_cache: dict[str, FlagReranker] = {}

def get_model(model_name: str) -> FlagReranker:
    """모델 로드 (캐싱)"""
    if model_name not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"unsupported model: {model_name}")
    
    if model_name not in model_cache:
        logger.info(f"Loading model: {MODEL_MAP[model_name]}")
        try:
            model_cache[model_name] = FlagReranker(
                MODEL_MAP[model_name],
                use_fp16=True,
                device=device
            )
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    return model_cache[model_name]

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """문서 재순위화 엔드포인트"""
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents cannot be empty")
        
        if len(request.documents) > 1000:
            raise HTTPException(status_code=400, detail="too many documents (max 1000)")
        
        model = get_model(request.model)
        
        # 쿼리-문서 쌍 생성
        pairs = [[request.query, doc] for doc in request.documents]
        
        # 배치로 한 번에 스코어 계산 (효율적!)
        scores = model.compute_score(pairs)
        
        # 단일 문서인 경우 리스트로 변환
        if isinstance(scores, float):
            scores = [scores]
        
        # 결과 생성
        results = [
            RerankResult(
                index=i,
                relevance_score=float(score),
                document=doc
            )
            for i, (doc, score) in enumerate(zip(request.documents, scores))
        ]
        
        # 점수 기준 내림차순 정렬
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # top_n 제한
        if request.top_n is not None and request.top_n > 0:
            results = results[:request.top_n]
        
        return RerankResponse(
            model=request.model,
            results=results,
            usage={
                "total_documents": len(request.documents),
                "reranked_documents": len(results)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reranking error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 워밍업"""
    logger.info("Starting BGE Reranker Server...")
    try:
        model = get_model("bge-reranker-v2-m3")
        # 워밍업
        _ = model.compute_score([["warmup query", "warmup document"]])
        logger.info("Model warmup completed")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")

@app.get("/v1/models")
async def list_models():
    """지원 모델 목록"""
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
    """헬스체크"""
    health_info = {
        "status": "healthy",
        "device": device,
        "models": list(MODEL_MAP.keys()),
        "models_loaded": list(model_cache.keys()),
    }
    
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
    
    parser = argparse.ArgumentParser(description="BGE Reranker Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8011, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
