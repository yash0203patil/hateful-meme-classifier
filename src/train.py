import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from src.models import get_model, count_params
from src.data import make_dataloaders
from src.utils import set_seed, AverageMeter, save_checkpoint

# ============ CONFIG ============
CONFIG = {
    "model_type": "image",          # image | text | fusion
    "freeze_encoders": False,         # Start frozen (unfreeze later if needed)
    "use_ocr": False,                 # Use OCR text (set False for ablation 4.b)
    "batch_size": 32,
    "num_workers": 4,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "epochs": 20,
    "warmup_epochs": 2,
    "patience": 5,                   # Early stopping patience
    "ckpt_dir": "checkpoints",
    "data_dir": "data",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, epoch):
    """Single training epoch with AMP (mixed precision)"""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}")
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).unsqueeze(1)  # [B,1]
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            if CONFIG["model_type"] == "image":
                logits = model(pixel_values=images)
            elif CONFIG["model_type"] == "text":
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:  # fusion
                logits = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(logits, labels)
        
        # Backward with AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # Metrics
        preds = (torch.sigmoid(logits).squeeze() >= 0.5).float()
        acc = accuracy_score(labels.cpu(), preds.cpu())
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.2%}")
    
    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation epoch - compute loss + ROC-AUC"""
    model.eval()
    loss_meter = AverageMeter()
    all_labels, all_logits = [], []
    
    pbar = tqdm(loader, desc="Validating")
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).unsqueeze(1)
        
        with autocast():
            if CONFIG["model_type"] == "image":
                logits = model(pixel_values=images)
            elif CONFIG["model_type"] == "text":
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(logits, labels)
        
        loss_meter.update(loss.item(), images.size(0))
        all_labels.append(labels.cpu())
        all_logits.append(torch.sigmoid(logits).cpu())
    
    # Aggregate metrics
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_logits).numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred) * 100
    try:
        auc = roc_auc_score(y_true, y_prob) * 100
    except:
        auc = 50.0
    
    return loss_meter.avg, acc, auc

def main():
    # Setup
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)
    
    ocr_status = "with OCR" if CONFIG["use_ocr"] else "caption-only"
    print(f"?? Training {CONFIG['model_type']} model ({ocr_status}) on {device}")
    print(f"?? Config: batch={CONFIG['batch_size']}, lr={CONFIG['lr']}, epochs={CONFIG['epochs']}")
    
    # Data
    train_loader, val_loader, test_loader, pos_weight = make_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        use_ocr=CONFIG["use_ocr"],
        seed=CONFIG["seed"]
    )
    print(f"? Data loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    
    # Model
    model = get_model(
        model_type=CONFIG["model_type"],
        freeze_encoders=CONFIG["freeze_encoders"],
        device=device
    )
    print(f"?? Model params: {count_params(model):,} trainable")
    
    # Loss + Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Scheduler (cosine with warmup)
    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = len(train_loader) * CONFIG["warmup_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # AMP scaler
    scaler = GradScaler()
    
    # Training loop
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    metrics_log = []
    
    start_time = time.time()
    
    for epoch in range(CONFIG["epochs"]):
        t0 = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device, epoch
        )
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        
        t1 = time.time()
        epoch_time = t1 - t0
        
        print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% AUC: {val_auc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # Log metrics
        metrics_log.append({
            "epoch": epoch+1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc*100, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_auc": round(val_auc, 2),
            "time": round(epoch_time, 1)
        })
        
        # Early stopping + checkpoint (based on ROC-AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            save_path = os.path.join(CONFIG["ckpt_dir"], f"best_{CONFIG['model_type']}.pt")
            save_checkpoint(model, optimizer, epoch+1, val_auc, save_path)
            print(f"?? New best AUC {val_auc:.2f}% ? saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"??  Early stopping at epoch {epoch+1} (no AUC improvement for {patience_counter} epochs)")
                break
    
    total_time = time.time() - start_time
    
    # Save training log
    log_path = os.path.join(CONFIG["ckpt_dir"], f"train_log_{CONFIG['model_type']}.json")
    with open(log_path, 'w') as f:
        json.dump({
            "config": CONFIG,
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "total_time_hours": round(total_time/3600, 2),
            "metrics": metrics_log
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"? Training complete!")
    print(f"?? Best Val AUC: {best_auc:.2f}% (epoch {best_epoch})")
    print(f"??  Total time: {total_time/3600:.2f} hours")
    print(f"?? Checkpoint: checkpoints/best_{CONFIG['model_type']}.pt")
    print(f"{'='*60}")
    
    return model, best_auc

if __name__ == "__main__":
    main()