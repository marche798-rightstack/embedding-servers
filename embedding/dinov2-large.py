import asyncio
import base64
import io
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor, AutoModel

# --- 모델 및 장치 설정 ---
MODEL_NAME = "facebook/dinov2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 로드 (서버 시작 시 한 번만 실행)
    print(f"🚀 Loading {MODEL_NAME} on {DEVICE}...")
    app.state.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    app.state.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    app.state.model.eval()
    app.state.http_client = httpx.AsyncClient()
    yield
    # 자원 해제
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

# --- 요청/응답 스키마 (OpenAI 규격 최적화) ---
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "dinov2-large"
    normalize: Optional[bool] = True
    # 추가: OpenAI 규격에 있는 필드
    user: Optional[str] = None

# --- 이미지 로더 (비동기 및 예외 처리 강화) ---
async def fetch_image(client: httpx.AsyncClient, source: str) -> Image.Image:
    try:
        if source.startswith(('http://', 'https://')):
            resp = await client.get(source, timeout=10.0)
            resp.raise_for_status()
            content = resp.content
        elif source.startswith('data:image'):
            content = base64.b64decode(source.split(',', 1)[1])
        else:
            # 순수 base64 처리
            content = base64.b64decode(source)
        
        return Image.open(io.BytesIO(content)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Image load error ({source[:20]}...): {str(e)}")

# --- 메인 엔드포인트 ---
@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    inputs_raw = [request.input] if isinstance(request.input, str) else request.input
    
    if not inputs_raw:
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        # 1. 비동기 병렬 이미지 로드
        image_tasks = [fetch_image(app.state.http_client, src) for src in inputs_raw]
        images = await asyncio.gather(*image_tasks)
        
        # 2. 배치 전처리
        inputs = app.state.processor(images=images, return_tensors="pt").to(DEVICE)
        
        # 3. 모델 추론 (배치 처리)
        with torch.no_grad():
            outputs = app.state.model(**inputs)
            # CLS 토큰 사용 (DINOv2 표준)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            if request.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 4. 결과 포맷팅 (OpenAI 규격 준수)
        embeddings_list = embeddings.cpu().numpy().tolist()
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": emb
            }
            for i, emb in enumerate(embeddings_list)
        ]
        
        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": len(inputs_raw), # 이미지당 1토큰으로 계산하거나 패치 수 적용
                "total_tokens": len(inputs_raw)
            }
        }

    except ValueError as ve:
        # OpenAI 스타일 에러 응답
        return {"error": {"message": str(ve), "type": "invalid_request_error"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)