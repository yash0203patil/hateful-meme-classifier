import os
import sys
import json
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Optional, Tuple
from torchvision import transforms
from transformers import AutoTokenizer, CLIPImageProcessor

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import FusionModel, ImageOnlyModel, TextOnlyModel
from src.utils import set_seed

# ============ CONFIG ============
INFERENCE_CONFIG = {
    "ckpt_path": "checkpoints/best_fusion.pt",
    "model_type": "fusion",          # fusion | image | text
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "img_model": "openai/clip-vit-base-patch32",
    "max_text_len": 64,
    "img_size": 224,
    "threshold": 0.5,
    "use_ocr": True,
    "ocr_cache_path": "data/ocr_cache.json",
    "seed": 42
}

# ============ GLOBAL STATE ============
_model = None
_tokenizer = None
_image_transform = None
_device = None
_ocr_cache = None


# ============ MODEL LOADING ============
def load_model(
    ckpt_path: str = None,
    model_type: str = None,
    device: str = None
) -> nn.Module:

    global _model, _device
    
    if _model is not None:
        return _model
    
    # Set defaults from config
    ckpt_path = ckpt_path or INFERENCE_CONFIG["ckpt_path"]
    model_type = model_type or INFERENCE_CONFIG["model_type"]
    device = device or INFERENCE_CONFIG["device"]
    
    _device = torch.device(device)
    
    # Select model class
    if model_type == "image":
        model = ImageOnlyModel(freeze_encoder=True)
    elif model_type == "text":
        model = TextOnlyModel(freeze_encoder=True)
    else:  # fusion
        model = FusionModel(freeze_encoders=True)
    
    # Load checkpoint
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=_device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded: {ckpt_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}, Val AUC: {checkpoint.get('metric', 'N/A'):.2f}%")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    model = model.to(_device)
    model.eval()
    
    _model = model
    return model


# ============ TOKENIZER LOADING ============
def load_tokenizer(text_model: str = None) -> AutoTokenizer:
    """Load tokenizer (singleton pattern - loads once)"""
    global _tokenizer
    
    if _tokenizer is not None:
        return _tokenizer
    
    text_model = text_model or INFERENCE_CONFIG["text_model"]
    _tokenizer = AutoTokenizer.from_pretrained(text_model)
    
    return _tokenizer


# ============ IMAGE PREPROCESSING ============
def load_image_transform(img_size: int = None) -> transforms.Compose:
    """Load image transform (singleton pattern - loads once)"""
    global _image_transform
    
    if _image_transform is not None:
        return _image_transform
    
    img_size = img_size or INFERENCE_CONFIG["img_size"]
    
    _image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return _image_transform


# ============ OCR CACHE LOADING ============
def load_ocr_cache(cache_path: str = None) -> Dict[str, str]:
    """Load OCR cache (singleton pattern - loads once)"""
    global _ocr_cache
    
    if _ocr_cache is not None:
        return _ocr_cache
    
    cache_path = cache_path or INFERENCE_CONFIG["ocr_cache_path"]
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            _ocr_cache = json.load(f)
        print(f"✅ OCR cache loaded: {len(_ocr_cache)} entries")
    else:
        _ocr_cache = {}
        print("⚠️  No OCR cache found. Using caption-only.")
    
    return _ocr_cache


# ============ IMAGE LOADING & PREPROCESSING ============
def load_and_preprocess_image(
    image_path: str,
    img_size: int = None
) -> torch.Tensor:

    transform = load_image_transform(img_size)
    
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    
    return img_tensor


# ============ TEXT PREPROCESSING ============
def build_text_input(
    caption: str,
    image_path: str = None,
    use_ocr: bool = None
) -> str:

    use_ocr = use_ocr if use_ocr is not None else INFERENCE_CONFIG["use_ocr"]
    
    if not use_ocr or image_path is None:
        return caption or ""
    
    # Load OCR cache
    ocr_cache = load_ocr_cache()
    
    # Extract image ID from path
    img_id = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')
    ocr_text = ocr_cache.get(img_id, "")
    
    if ocr_text.strip():
        return f"{caption} [OCR] {ocr_text.strip()}"
    return caption or ""


def tokenize_text(
    text: str,
    max_length: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    tokenizer = load_tokenizer()
    max_length = max_length or INFERENCE_CONFIG["max_text_len"]
    
    enc = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    
    return input_ids, attention_mask


# ============ INFERENCE FUNCTION ============
@torch.no_grad()
def predict(
    image_path: str = None,
    caption: str = "",
    use_ocr: bool = None,
    threshold: float = None,
    model_type: str = None,
    return_features: bool = False
) -> Dict:

    # Load model
    model_type = model_type or INFERENCE_CONFIG["model_type"]
    model = load_model(model_type=model_type)
    threshold = threshold or INFERENCE_CONFIG["threshold"]
    
    # Prepare inputs
    image_tensor = None
    input_ids = None
    attention_mask = None
    text_input = ""
    
    # Load and preprocess image (for image/fusion models)
    if model_type in ["image", "fusion"] and image_path:
        image_tensor = load_and_preprocess_image(image_path)
        image_tensor = image_tensor.to(_device)
    
    # Build and tokenize text (for text/fusion models)
    if model_type in ["text", "fusion"]:
        text_input = build_text_input(caption, image_path, use_ocr)
        input_ids, attention_mask = tokenize_text(text_input)
        input_ids = input_ids.to(_device)
        attention_mask = attention_mask.to(_device)
    
    # Run inference
    if model_type == "image":
        logits = model(pixel_values=image_tensor)
    elif model_type == "text":
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    else:  # fusion
        logits = model(
            pixel_values=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Compute probability
    prob = torch.sigmoid(logits).item()
    label = "hateful" if prob >= threshold else "not_hateful"
    
    # Build result
    result = {
        "label": label,
        "confidence": round(prob, 4),
        "logit": round(logits.item(), 4),
        "threshold": threshold,
        "image_path": image_path,
        "caption": caption,
        "text_input": text_input,
        "model_type": model_type
    }
    
    if return_features:
        result["image_tensor_shape"] = image_tensor.shape if image_tensor is not None else None
        result["input_ids_shape"] = input_ids.shape if input_ids is not None else None
    
    return result


# ============ BATCH INFERENCE ============
@torch.no_grad()
def predict_batch(
    samples: list,
    batch_size: int = 32,
    **kwargs
) -> list:
    """
    Run inference on multiple samples
    
    Args:
        samples: List of dicts with keys: image_path, caption
        batch_size: Batch size for inference
        **kwargs: Passed to predict()
    
    Returns:
        List of prediction results
    """
    results = []
    
    for i, sample in enumerate(samples):
        result = predict(
            image_path=sample.get("image_path"),
            caption=sample.get("caption", ""),
            **kwargs
        )
        result["sample_id"] = sample.get("id", i)
        results.append(result)
    
    return results


# ============ RESET STATE ============
def reset_inference_state():
    """Reset all cached state (useful for testing)"""
    global _model, _tokenizer, _image_transform, _device, _ocr_cache
    _model = None
    _tokenizer = None
    _image_transform = None
    _device = None
    _ocr_cache = None
    print("✅ Inference state reset")


# ============ COMMAND-LINE INTERFACE ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Single-sample inference for Hateful Memes Classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--caption", type=str, default="", help="Caption text")
    parser.add_argument("--model-type", type=str, default="fusion", choices=["fusion", "image", "text"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--use-ocr", action="store_true", default=True, help="Include OCR text")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(INFERENCE_CONFIG["seed"])
    
    # Run inference
    result = predict(
        image_path=args.image,
        caption=args.caption,
        model_type=args.model_type,
        threshold=args.threshold,
        use_ocr=not args.no_ocr,
        ckpt_path=args.checkpoint
    )
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*60)
        print("🔮 INFERENCE RESULT")
        print("="*60)
        print(f"Image:        {result['image_path']}")
        print(f"Caption:      {result['caption']}")
        print(f"Text Input:   {result['text_input'][:100]}...")
        print(f"Model:        {result['model_type']}")
        print(f"Label:        {result['label']}")
        print(f"Confidence:   {result['confidence']:.4f}")
        print(f"Logit:        {result['logit']:.4f}")
        print(f"Threshold:    {result['threshold']}")
        print("="*60)