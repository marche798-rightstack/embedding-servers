from typing import List, Union, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import uvicorn, torch, base64, requests, numpy as np
from io import BytesIO

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 레지스트리: 노출할 이름 -> (type, HF id, canonical name)
MODEL_REGISTRY: Dict[str, dict] = {
    "bge-m3": {"type": "text", "hf": "BAAI/bge-m3", "canonical": "bge-m3"},
    "BAAI/bge-m3": {"type": "text", "hf": "BAAI/bge-m3", "canonical": "bge-m3"},
    "dinov2-large": {"type": "image", "hf": "facebook/dinov2-large", "canonical": "dinov2-large"},
    "facebook/dinov2-large": {"type": "image", "hf": "facebook/dinov2-large", "canonical": "dinov2-large"},
}

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str = Field(default="bge-m3")
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")
    dimensions: Optional[int] = Field(default=None, gt=0)
    user: Optional[str] = None  # 전달만 함

text_models: dict[str, SentenceTransformer] = {}
image_models: dict[str, tuple[AutoImageProcessor, AutoModel]] = {}

def get_model(info: dict):
    if info["type"] == "text":
        if info["canonical"] not in text_models:
            text_models[info["canonical"]] = SentenceTransformer(info["hf"], device=device)
        return text_models[info["canonical"]]
    if info["type"] == "image":
        if info["canonical"] not in image_models:
            proc = AutoImageProcessor.from_pretrained(info["hf"])
            mdl = AutoModel.from_pretrained(info["hf"]).to(device).eval()
            image_models[info["canonical"]] = (proc, mdl)
        return image_models[info["canonical"]]
    raise HTTPException(status_code=400, detail="unsupported model type")

def normalize_text_input(raw) -> List[str]:
    if isinstance(raw, str):
        if not raw.strip():
            raise HTTPException(status_code=400, detail="input cannot be empty")
        return [raw]
    if isinstance(raw, list):
        if not raw:
            raise HTTPException(status_code=400, detail="input list cannot be empty")
        if all(isinstance(x, int) for x in raw):
            return [" ".join(map(str, raw))]
        if all(isinstance(x, list) and x and all(isinstance(t, int) for t in x) for x in raw):
            return [" ".join(map(str, seq)) for seq in raw]
        return [str(t) for t in raw]
    return [str(raw)]

def load_image(item: str) -> Image.Image:
    if item.startswith("http://") or item.startswith("https://"):
        try:
            resp = requests.get(item, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            raise HTTPException(status_code=400, detail=f"failed to fetch image ({status}): {item}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to fetch image: {e}")
        return Image.open(BytesIO(resp.content)).convert("RGB")
    # base64 지원 (data: prefix는 제거)
    try:
        if item.startswith("data:"):
            item = item.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(item))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image input (url or base64 expected)")

def trim_dimensions(vec: np.ndarray, dim: Optional[int]) -> np.ndarray:
    return vec if dim is None else vec[:, :dim]

def encode_output(vectors: np.ndarray, encoding_format: str):
    if encoding_format == "float":
        return vectors.tolist()
    as_bytes = vectors.astype(np.float32).tobytes()
    return base64.b64encode(as_bytes).decode("ascii")

@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    info = MODEL_REGISTRY.get(req.model)
    if not info:
        raise HTTPException(status_code=400, detail=f"unsupported model: {req.model}")

    canonical = info["canonical"]

    # 입력 정규화 & 인코딩
    if info["type"] == "text":
        texts = normalize_text_input(req.input)
        if len(texts) > 2048:
            raise HTTPException(status_code=400, detail="too many inputs (max 2048)")
        model = get_model(info)
        embeddings = model.encode(
            texts, batch_size=min(max(8, len(texts)), 64), normalize_embeddings=True
        )
    else:  # image
        items = req.input if isinstance(req.input, list) else [req.input]
        if not items:
            raise HTTPException(status_code=400, detail="input list cannot be empty")
        if len(items) > 128:
            raise HTTPException(status_code=400, detail="too many images (max 128)")
        images = [load_image(str(it)) for it in items]
        processor, mdl = get_model(info)
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = mdl(**inputs)
            # CLS 토큰 벡터 사용 (dinov2)
            feats = outputs.last_hidden_state[:, 0, :]
            feats = torch.nn.functional.normalize(feats, dim=1)
            embeddings = feats.cpu().numpy()

    embeddings = trim_dimensions(np.array(embeddings), req.dimensions)
    encoded = encode_output(embeddings, req.encoding_format)

    data = []
    if req.encoding_format == "float":
        for i, emb in enumerate(encoded):
            data.append({"object": "embedding", "embedding": emb, "index": i})
    else:
        # base64는 하나의 문자열을 모든 index에 공유 (OpenAI 규격)
        for i in range(len(embeddings)):
            data.append({"object": "embedding", "embedding": encoded, "index": i})

    usage_tokens = sum(len(str(t).split()) for t in (texts if info["type"] == "text" else items))

    return {
        "object": "list",
        "data": data,
        "model": canonical,
        "usage": {"prompt_tokens": usage_tokens, "total_tokens": usage_tokens},
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)


