# src/precompute_ocr.py
import os, json, torch
from PIL import Image
from tqdm import tqdm
import easyocr

def precompute_ocr(data_dir='data', output_file='data/ocr_cache.json'):
    """Pre-compute OCR for all images once and save to JSON"""
    
    # Initialize OCR reader (GPU accelerated)
    print("?? Initializing EasyOCR (GPU enabled)...")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
    
    # Collect all image paths from all splits
    image_paths = set()
    for jsonl in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:
        path = os.path.join(data_dir, jsonl)
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    # Map image path relative to data_dir
                    img_path = os.path.join(data_dir, item['img'])
                    image_paths.add(img_path)
    
    print(f"?? Found {len(image_paths)} unique images to process...")
    ocr_cache = {}
    
    for img_path in tqdm(image_paths, desc="Extracting OCR"):
        if not os.path.exists(img_path):
            continue
            
        # Extract ID from path (e.g., "data/img/42953.png" -> "42953")
        img_id = os.path.basename(img_path).replace('.png', '')
        
        try:
            result = reader.readtext(img_path, detail=0)
            ocr_text = " ".join(result).strip()
        except Exception as e:
            ocr_text = ""
        
        ocr_cache[img_id] = ocr_text
    
    # Save cache
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(ocr_cache, f, indent=2)
    
    print(f"\n OCR cache saved to {output_file}")
    print(f"   Total images: {len(ocr_cache)}")
    print(f"   Images with text: {sum(1 for v in ocr_cache.values() if v)}")
    
    return ocr_cache

if __name__ == "__main__":
    precompute_ocr()