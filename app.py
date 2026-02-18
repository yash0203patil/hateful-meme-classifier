import os
import sys
import io
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, CLIPImageProcessor
from torchvision import transforms
import uvicorn
from src.models import FusionModel
from src.utils import set_seed

# ============ CONFIG ============
CONFIG = {
    "ckpt_path": "checkpoints/best_fusion.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "img_model": "openai/clip-vit-base-patch32",
    "max_text_len": 64,
    "img_size": 224,
    "threshold": 0.5,
    "max_file_size": 10 * 1024 * 1024,
    "allowed_extensions": {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
}


set_seed(42)
app = FastAPI(
    title="Hateful Memes Classifier",
    description="Multimodal classifier for detecting hateful memes (Image + Text)",
    version="1.0.0"
)

# Global model instance
model = None
tokenizer = None
device = None
image_transform = None

def load_model():
    """Load trained model checkpoint (called once)"""
    global model, tokenizer, device, image_transform
    
    if model is None:
        device = torch.device(CONFIG["device"])
        
        # Load model
        model = FusionModel(freeze_encoders=True)
        model = model.to(device)
        
        if os.path.exists(CONFIG["ckpt_path"]):
            checkpoint = torch.load(CONFIG["ckpt_path"], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"? Model loaded: {CONFIG['ckpt_path']}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {CONFIG['ckpt_path']}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])
        
        # Create image transform (proper PIL?Tensor conversion)
        image_transform = transforms.Compose([
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),  # ? Properly converts PIL to [0,1] tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        model.eval()

def validate_image(image: UploadFile) -> Image.Image:
    """Validate and load image file"""
    ext = os.path.splitext(image.filename)[1].lower()
    
    if ext and ext not in CONFIG["allowed_extensions"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {CONFIG['allowed_extensions']}"
        )
    
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    image_bytes = image.file.read()
    if len(image_bytes) > CONFIG["max_file_size"]:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {CONFIG['max_file_size'] / 1024 / 1024:.0f}MB"
        )
    
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    
    return img

def process_image(img: Image.Image) -> torch.Tensor:
    img_tensor = image_transform(img)  # [C, H, W], normalized
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
    return img_tensor

def process_text(caption: str) -> tuple:
    """Tokenize text input"""
    enc = tokenizer(
        caption if caption else "",
        padding='max_length',
        truncation=True,
        max_length=CONFIG["max_text_len"],
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    return input_ids, attention_mask

def predict(image_tensor: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    """Run model inference"""
    with torch.no_grad():
        logits = model(
            pixel_values=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        prob = torch.sigmoid(logits).item()
    
    label = "hateful" if prob >= CONFIG["threshold"] else "not_hateful"
    
    return {
        "label": label,
        "confidence": round(prob, 4),
        "threshold": CONFIG["threshold"]
    }

# ============ ENDPOINTS ============
@app.get("/")
async def root():
    return {
        "service": "Hateful Memes Classifier",
        "version": "1.0.0",
        "endpoints": {"/": "Info", "/health": "Health", "/info": "Model info", "/predict": "POST - Predict"}
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device) if device else "not_loaded",
        "model_loaded": model is not None
    }

@app.get("/info")
async def model_info():
    return {
        "model_type": "Fusion (Image + Text)",
        "image_encoder": "CLIP ViT-B/32",
        "text_encoder": "MiniLM-L6-v2",
        "fusion": "Late concatenation + MLP",
        "checkpoint": CONFIG["ckpt_path"],
        "device": str(device) if device else "not_loaded",
        "threshold": CONFIG["threshold"]
    }

@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(..., description="Meme image file"),
    caption: str = Form(None, description="Optional caption text")
):
    """Classify meme as hateful or not_hateful"""
    load_model()
    
    try:
        # Validate and load image
        img = validate_image(image)
        
        # Preprocess
        image_tensor = process_image(img)  # ? Fixed conversion
        input_ids, attention_mask = process_text(caption)
        
        # Predict
        result = predict(image_tensor, input_ids, attention_mask)
        
        # Add metadata
        result["image_name"] = image.filename
        result["caption_provided"] = caption is not None and len(caption) > 0
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")  # Log for debugging
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============ STARTUP ============
@app.on_event("startup")
async def startup_event():
    print("Starting Hateful Memes Classifier API...")
    load_model()
    print("API ready")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")