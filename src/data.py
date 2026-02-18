import os
import json
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional


IMG_SIZE = 224
MAX_TEXT_LEN = 64
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OCR_CACHE_FILE = "ocr_cache.json"
SPLITS_FILE = "splits.json"

# Global OCR cache (loaded once per process)
OCR_CACHE: Optional[Dict[str, str]] = None


# ============ OCR CACHE UTILS ============
def load_ocr_cache(data_dir: str = "data") -> Dict[str, str]:
    """Load pre-computed OCR cache from JSON file"""
    global OCR_CACHE
    if OCR_CACHE is None:
        cache_path = os.path.join(data_dir, OCR_CACHE_FILE)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                OCR_CACHE = json.load(f)
            print(f"? Loaded OCR cache: {len(OCR_CACHE)} entries")
        else:
            OCR_CACHE = {}
            print("??  No OCR cache found. Using caption-only mode.")
            print("   Run: python3 src/precompute_ocr.py")
    return OCR_CACHE


def extract_ocr_from_cache(img_path: str, use_ocr: bool = True) -> str:
    """Extract OCR text from pre-computed cache (fast!)"""
    if not use_ocr:
        return ""
    
    cache = load_ocr_cache(os.path.dirname(img_path))
    if not cache:
        return ""
    
    # Extract image ID from path (e.g., "data/img/42953.png" -> "42953")
    img_id = os.path.basename(img_path).replace('.png', '')
    return cache.get(img_id, "")


def build_text_input(caption: str, ocr_text: str, use_ocr: bool = True) -> str:
    """
    Build text modality as: caption + " [OCR] " + ocr_text
    Fallback to caption-only if OCR fails or disabled (for ablation)
    
    Requirement 1.3: caption + " [OCR] " + ocr_text
    """
    if use_ocr and ocr_text.strip():
        return f"{caption} [OCR] {ocr_text.strip()}"
    return caption


# ============ DATASET CLASS ============
class HatefulMemesDataset(Dataset):
    """
    Multimodal dataset for hateful memes (image + text)
    
    Requirement 1.1: Each sample contains image, caption, and label
    Requirement 1.3: Text = caption + OCR
    """
    
    def __init__(
        self,
        data_dir: str,
        split_file: str,
        mode: str = 'train',
        tokenizer = None,
        use_ocr: bool = True,
        augment: bool = False
    ):
        """
        Args:
            data_dir: Path to data directory (contains img/, jsonl files)
            split_file: Path to splits.json
            mode: 'train', 'val', or 'test'
            tokenizer: HuggingFace tokenizer
            use_ocr: Whether to use OCR text (disable for ablation 4.b)
            augment: Whether to apply train augmentations
        """
        self.data_dir = data_dir
        self.mode = mode
        self.use_ocr = use_ocr
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(TEXT_MODEL)
        
        # Load split indices
        with open(split_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        self.ids = splits.get(mode, [])
        
        if len(self.ids) == 0:
            raise ValueError(f"No samples found for mode='{mode}' in {split_file}")
        
        # Load metadata from all JSONL files
        self.samples = {}
        for jl_file in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:
            path = os.path.join(data_dir, jl_file)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        self.samples[item['id']] = item
        
        # Image transforms
        # Requirement 1.4: Basic augmentations on train only
        if augment and mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Load OCR cache (if enabled)
        if use_ocr:
            load_ocr_cache(data_dir)
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.ids[idx]
        item = self.samples[sid]
        
        # Load image
        img_path = os.path.join(self.data_dir, item['img'])
        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)
        
        # Build text input (caption + optional OCR)
        # Requirement 1.3: caption + " [OCR] " + ocr_text
        caption = item.get('text', '')
        ocr_text = extract_ocr_from_cache(img_path, use_ocr=self.use_ocr)
        text_input = build_text_input(caption, ocr_text, use_ocr=self.use_ocr)
        
        # Tokenize text
        enc = self.tokenizer(
            text_input,
            padding='max_length',
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors='pt'
        )
        
        # Get label (float for BCEWithLogitsLoss)
        # Handle test set where label may be missing
        label = item.get('label', None)
        if label is not None:
            label = torch.tensor(float(label), dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)  # Placeholder for test set
        
        return {
            'image': image,
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': label,
            'id': sid,
            'text': text_input  # For debugging/analysis
        }


# ============ SPLIT UTILS ============
def create_stratified_splits(
    data_dir: str,
    output_file: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, List[int]]:
    random.seed(seed)
    np.random.seed(seed)
    
    # Load samples ONLY from files that have labels
    # (test.jsonl typically doesn't have labels in competition datasets)
    samples = []
    labeled_files = ['train.jsonl', 'dev.jsonl']
    
    for jsonl in labeled_files:
        path = os.path.join(data_dir, jsonl)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    # Only include samples that have a 'label' key
                    if 'label' in item:
                        samples.append(item)
    
    if len(samples) == 0:
        raise ValueError("No labeled samples found! Check your JSONL files.")
    
    print(f"?? Total labeled samples loaded: {len(samples)}")
    
    # Separate by label for stratification
    zeros = [s['id'] for s in samples if s['label'] == 0]
    ones = [s['id'] for s in samples if s['label'] == 1]
    
    print(f"   Class 0 (Not Hateful): {len(zeros)}")
    print(f"   Class 1 (Hateful):     {len(ones)}")
    
    def stratified_split(ids: List[int], train_r: float, val_r: float) -> Tuple[List[int], List[int], List[int]]:
        """Split a list of IDs while preserving class balance"""
        ids_copy = ids.copy()
        random.shuffle(ids_copy)
        n = len(ids_copy)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        return ids_copy[:train_end], ids_copy[train_end:val_end], ids_copy[val_end:]
    
    # Split each class separately (stratification)
    train_0, val_0, test_0 = stratified_split(zeros, train_ratio, val_ratio)
    train_1, val_1, test_1 = stratified_split(ones, train_ratio, val_ratio)
    
    # Combine splits
    splits = {
        'train': sorted(train_0 + train_1),
        'val': sorted(val_0 + val_1),
        'test': sorted(test_0 + test_1)
    }
    
    # Save splits to JSON (Requirement 1.5)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    
    # Print split summary
    print(f"\n? Splits saved to {output_file}")
    for mode in ['train', 'val', 'test']:
        total = len(splits[mode])
        hateful = sum(1 for sid in splits[mode] if sid in [s['id'] for s in samples if s['label'] == 1])
        pct = hateful / total * 100 if total > 0 else 0
        print(f"   {mode:5s}: {total:5d} samples ({hateful:4d} hateful, {pct:5.1f}%)")
    
    return splits


# ============ DATALOADER FACTORY ============
def make_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    use_ocr: bool = True,
    seed: int = 42,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    # Set seeds for reproducibility (Requirement 5.1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create splits if they don't exist
    split_file = os.path.join(data_dir, SPLITS_FILE)
    if not os.path.exists(split_file):
        print("?? Creating stratified splits...")
        create_stratified_splits(data_dir, split_file, seed=seed)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    
    # Create datasets
    train_ds = HatefulMemesDataset(
        data_dir=data_dir,
        split_file=split_file,
        mode='train',
        tokenizer=tokenizer,
        use_ocr=use_ocr,
        augment=True  # Train augmentations
    )
    
    val_ds = HatefulMemesDataset(
        data_dir=data_dir,
        split_file=split_file,
        mode='val',
        tokenizer=tokenizer,
        use_ocr=use_ocr,
        augment=False
    )
    
    test_ds = HatefulMemesDataset(
        data_dir=data_dir,
        split_file=split_file,
        mode='test',
        tokenizer=tokenizer,
        use_ocr=use_ocr,
        augment=False
    )
    
    # Calculate pos_weight for BCEWithLogitsLoss (Requirement 2.4)
    # pos_weight = neg_count / pos_count
    train_labels = [train_ds.samples[sid]['label'] for sid in train_ds.ids if 'label' in train_ds.samples[sid]]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    
    print(f"\n??  Class imbalance - pos_weight: {pos_weight.item():.4f}")
    print(f"   Positive samples: {pos_count} ({pos_count/len(train_labels)*100:.1f}%)")
    print(f"   Negative samples: {neg_count} ({neg_count/len(train_labels)*100:.1f}%)")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, pos_weight


# ============ QUICK TEST ============
if __name__ == "__main__":
    """Quick sanity check for data pipeline"""
    print("?? Testing data pipeline...\n")
    
    # Create splits
    create_stratified_splits("data", "data/splits.json", seed=42)
    
    # Create dataloaders with OCR enabled (main model)
    train_loader, val_loader, test_loader, pos_weight = make_dataloaders(
        data_dir="data",
        batch_size=4,
        num_workers=2,
        use_ocr=True  # Set to False for caption-only ablation (Requirement 4.b)
    )
    
    print(f"\n?? DataLoader Summary:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    # Check one batch
    batch = next(iter(train_loader))
    print(f"\n?? One Batch Shapes:")
    print(f"   images:         {batch['image'].shape}")
    print(f"   input_ids:      {batch['input_ids'].shape}")
    print(f"   attention_mask: {batch['attention_mask'].shape}")
    print(f"   labels:         {batch['label'].shape}")
    print(f"   labels values:  {batch['label'].tolist()}")
    
    # Check text input (first sample) - verify OCR integration
    print(f"\n?? Sample Text Input:")
    print(f"   {batch['text'][0][:150]}...")
    
    # Verify OCR is being used
    if '[OCR]' in batch['text'][0]:
        print("   ? OCR text detected in input")
    else:
        print("   ??  No OCR text (check ocr_cache.json)")
    
    print("\n? Data pipeline test passed!")