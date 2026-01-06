from typing import List, Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from PIL import Image
import io
import base64

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load DINOv3 Large embedding model once at startup
print("Loading DINOv3 Large embedding model")
model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()
print(f"Model loaded! Embedding dimension: 1024")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    normalize: Optional[bool] = True


def load_image_input(image_input: str) -> Image.Image:
    """Load image from URL, file path, or base64"""
    # URL이나 파일 경로 - transformers의 load_image 사용
    if image_input.startswith(('http://', 'https://', '/')):
        return load_image(image_input)
    
    # base64 처리
    if image_input.startswith('data:image'):
        image_input = image_input.split(',', 1)[1]
    image_bytes = base64.b64decode(image_input)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    raw = request.input
    
    if isinstance(raw, str):
        image_inputs = [raw]
    else:
        image_inputs = raw
    
    embeddings = []
    
    for image_input in image_inputs:
        # URL, 파일경로, base64 모두 지원
        image = load_image_input(image_input)
        
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            
            if request.normalize:
                embedding = embedding / embedding.norm()
            
            embeddings.append(embedding.cpu().numpy())
    
    data = [
        {"object": "embedding", "embedding": emb.tolist(), "index": i}
        for i, emb in enumerate(embeddings)
    ]
    
    return {
        "object": "list",
        "data": data,
        "model": request.model or "dinov3-large",
        "usage": {"prompt_tokens": len(image_inputs), "total_tokens": len(image_inputs)},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)




