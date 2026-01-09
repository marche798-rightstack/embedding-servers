from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from FlagEmbedding import FlagReranker
import uvicorn
import torch
import logging
import uuid
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BGE Reranker Server",
    description="Cohere-compatible reranking API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# 지원 모델 매핑
MODEL_MAP = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
}

class RerankRequest(BaseModel):
    """Cohere Rerank API 호환 요청"""
    query: str = Field(..., description="검색 쿼리")
    documents: List[str] = Field(..., min_items=1, max_items=1000, description="재순위화할 문서 리스트")
    model: str = Field(default="bge-reranker-v2-m3", description="사용할 모델")
    top_n: Optional[int] = Field(default=None, gt=0, description="상위 N개 결과만 반환")
    return_documents: bool = Field(default=False, description="결과에 문서 텍스트 포함 여부")
    max_chunks_per_doc: Optional[int] = Field(default=None, description="문서당 최대 청크 수 (미구현)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "documents": [
                    "Paris is the capital of France and its largest city.",
                    "Berlin is the capital and largest city of Germany.",
                    "London is the capital of England and the United Kingdom."
                ],
                "model": "bge-reranker-v2-m3",
                "top_n": 3,
                "return_documents": True
            }
        }

class RerankResult(BaseModel):
    """Cohere 호환 재순위화 결과"""
    index: int = Field(..., description="원본 documents 배열에서의 인덱스")
    relevance_score: float = Field(..., description="관련성 점수 (높을수록 관련성 높음)")
    document: Optional[str] = Field(default=None, description="문서 텍스트 (return_documents=True일 때만)")

class RerankMeta(BaseModel):
    """Cohere 호환 메타데이터"""
    api_version: dict = {"version": "1"}
    billed_units: Optional[dict] = None

class RerankResponse(BaseModel):
    """Cohere Rerank API 호환 응답"""
    id: str = Field(..., description="요청 고유 ID")
    results: List[RerankResult] = Field(..., description="재순위화된 결과 (관련성 내림차순)")
    meta: RerankMeta = Field(..., description="API 메타데이터")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "results": [
                    {
                        "index": 0,
                        "relevance_score": 0.9876,
                        "document": "Paris is the capital of France and its largest city."
                    },
                    {
                        "index": 2,
                        "relevance_score": 0.3456,
                        "document": "London is the capital of England and the United Kingdom."
                    }
                ],
                "meta": {
                    "api_version": {"version": "1"},
                    "billed_units": {"search_units": 3}
                }
            }
        }

model_cache: dict[str, FlagReranker] = {}

def get_model(model_name: str) -> FlagReranker:
    """모델 로드 및 캐싱"""
    if model_name not in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model_name}. Available models: {list(MODEL_MAP.keys())}"
        )
    
    if model_name not in model_cache:
        logger.info(f"Loading model: {MODEL_MAP[model_name]}")
        try:
            # FP16은 GPU에서만 사용 (CPU에서는 지원 안 함)
            use_fp16 = (device == "cuda")
            model_cache[model_name] = FlagReranker(
                MODEL_MAP[model_name],
                use_fp16=use_fp16,
                device=device
            )
            logger.info(f"Model loaded successfully on {device} (FP16: {use_fp16})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Model loading failed: {str(e)}"
            )
    
    return model_cache[model_name]

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Cohere 호환 문서 재순위화 엔드포인트
    
    주어진 쿼리에 대해 문서들의 관련성을 평가하고 점수순으로 정렬합니다.
    """
    try:
        # 입력 검증
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents cannot be empty")
        
        if len(request.documents) > 1000:
            raise HTTPException(status_code=400, detail="too many documents (max 1000)")
        
        # 모델 로드
        model = get_model(request.model)
        
        # 쿼리-문서 쌍 생성
        pairs = [[request.query, doc] for doc in request.documents]
        
        # 배치로 한 번에 스코어 계산 (효율적!)
        logger.info(f"Reranking {len(request.documents)} documents")
        scores = model.compute_score(pairs)
        
        # 단일 문서인 경우 리스트로 변환
        if isinstance(scores, float):
            scores = [scores]
        
        # 결과 생성
        results = []
        for i, score in enumerate(scores):
            result_data = {
                "index": i,
                "relevance_score": float(score)
            }
            # return_documents=True일 때만 문서 포함 (Cohere 표준)
            if request.return_documents:
                result_data["document"] = request.documents[i]
            
            results.append(RerankResult(**result_data))
        
        # 관련성 점수 기준 내림차순 정렬
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # top_n 제한
        original_count = len(results)
        if request.top_n is not None and request.top_n > 0:
            results = results[:request.top_n]
        
        # Cohere 호환 응답 생성
        response = RerankResponse(
            id=str(uuid.uuid4()),  # 고유 요청 ID
            results=results,
            meta=RerankMeta(
                api_version={"version": "1"},
                billed_units={
                    "search_units": original_count  # 처리한 문서 수
                }
            )
        )
        
        logger.info(f"Reranking completed: {original_count} docs -> top {len(results)}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reranking error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 워밍업"""
    logger.info("Starting BGE Reranker Server (Cohere-compatible)...")
    try:
        # 기본 모델 미리 로드
        model = get_model("bge-reranker-v2-m3")
        # 워밍업 추론
        _ = model.compute_score([["warmup query", "warmup document"]])
        logger.info("Model warmup completed")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")

@app.get("/v1/models")
async def list_models():
    """사용 가능한 모델 목록 반환"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "BAAI"
            }
            for model_id in MODEL_MAP.keys()
        ]
    }

@app.get("/health")
async def health():
    """
    헬스체크 엔드포인트
    
    서버 상태 및 GPU 정보 반환
    """
    health_info = {
        "status": "healthy",
        "device": device,
        "models_available": list(MODEL_MAP.keys()),
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

@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "BGE Reranker Server",
        "version": "1.0.0",
        "compatibility": "Cohere Rerank API",
        "endpoints": {
            "rerank": "/v1/rerank",
            "models": "/v1/models",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BGE Reranker Server (Cohere-compatible)")
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
