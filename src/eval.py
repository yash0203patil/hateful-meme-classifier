import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt


from src.models import get_model
from src.data import make_dataloaders
from src.utils import set_seed

# ============ CONFIG ============
CONFIG = {
    "model_type": "image",
    "batch_size": 32,
    "ckpt_path": "checkpoints/best_image.pt",
    "data_dir": "data",
    "output_dir": "results",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "use_ocr": True
}

@torch.no_grad()
def evaluate(model, loader, device):
    """Run inference on full dataset"""
    model.eval()
    all_labels, all_logits, all_ids = [], [], []
    
    pbar = tqdm(loader, desc="Evaluating")
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        ids = batch['id']
        
        if CONFIG["model_type"] == "image":
            logits = model(pixel_values=images)
        elif CONFIG["model_type"] == "text":
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
        
        all_labels.append(labels.cpu())
        all_logits.append(torch.sigmoid(logits).cpu())
        all_ids.extend(ids)
    
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_logits).numpy().flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    return y_true, y_prob, y_pred, all_ids

def compute_metrics(y_true, y_prob, y_pred):
    # Accuracy
    acc = accuracy_score(y_true, y_pred) * 100
    
    # Precision, Recall, F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    prec, rec, f1 = prec * 100, rec * 100, f1 * 100
    
    # ROC-AUC
    auc = roc_auc_score(y_true, y_prob) * 100
    
    # Confusion Matrix (% normalized by true class) (Requirement 3.3.b)
    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100
    tn_pct, fp_pct = cm[0, 0], cm[0, 1]
    fn_pct, tp_pct = cm[1, 0], cm[1, 1]
    
    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
        "roc_auc": round(auc, 2),
        "confusion_matrix_pct": {
            "TN%": round(tn_pct, 2),
            "FP%": round(fp_pct, 2),
            "FN%": round(fn_pct, 2),
            "TP%": round(tp_pct, 2)
        },
        "confusion_matrix_raw": cm.round(2).tolist()
    }

def plot_roc_curve(y_true, y_prob, output_path):
    """Generate ROC curve plot"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) * 100
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f}%)')
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"?? ROC curve saved to {output_path}")

def plot_confusion_matrix(cm, output_path):
    """Generate confusion matrix heatmap (%)"""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    
    # Add percentage labels
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]:.1f}%', ha='center', va='center',
                    color='darkblue' if cm[i, j] > 50 else 'white', fontsize=14)
    
    plt.xticks([0, 1], ['Not Hateful', 'Hateful'])
    plt.yticks([0, 1], ['Not Hateful', 'Hateful'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (%)')
    plt.colorbar(label='%')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"?? Confusion matrix saved to {output_path}")

def save_misclassified_examples(y_true, y_pred, y_prob, ids, output_path, top_n=10):
    """Save details of misclassified examples (Requirement 3.e)"""
    errors = []
    for true, pred, prob, sid in zip(y_true, y_pred, y_prob, ids):
        if true != pred:
            errors.append({
                "id": int(sid),
                "true_label": int(true),
                "pred_label": int(pred),
                "confidence": round(float(prob), 4),
                "error_type": "false_positive" if true == 0 else "false_negative"
            })
    
    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x['confidence'], reverse=(errors[0]['true_label'] == 0))
    
    with open(output_path, 'w') as f:
        json.dump(errors[:top_n], f, indent=2)
    
    print(f"? {len(errors)} misclassified examples saved to {output_path}")
    return errors[:top_n]

def main():
    # Setup
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    ocr_status = "with OCR" if CONFIG["use_ocr"] else "caption-only"
    print(f"?? Evaluating {CONFIG['model_type']} model ({ocr_status}) on test set...")
    print(f"?? Checkpoint: {CONFIG['ckpt_path']}")
    
    # Verify checkpoint exists
    if not os.path.exists(CONFIG["ckpt_path"]):
        print(f"? Checkpoint not found: {CONFIG['ckpt_path']}")
        print("   Run training first: python3 -m src.train")
        return
    
    # Load data (test set only)
    _, _, test_loader, _ = make_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        use_ocr=CONFIG["use_ocr"]
    )
    print(f"? Test samples: {len(test_loader.dataset)}")
    
    # Load model
    model = get_model(CONFIG["model_type"], freeze_encoders=True, device=device)
    checkpoint = torch.load(CONFIG["ckpt_path"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"?? Loaded checkpoint (epoch {checkpoint['epoch']}, val_auc {checkpoint['metric']:.2f}%)")
    
    # Run evaluation
    y_true, y_prob, y_pred, ids = evaluate(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_prob, y_pred)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS (%)")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall:    {metrics['recall']:.2f}%")
    print(f"F1 Score:  {metrics['f1']:.2f}%")
    print(f"ROC-AUC:   {metrics['roc_auc']:.2f}%")
    print("\nConfusion Matrix (% normalized by true class):")
    cm = metrics['confusion_matrix_raw']
    print(f"[[{cm[0][0]:6.2f}  {cm[0][1]:6.2f}]    [TN%  FP%]")
    print(f" [{cm[1][0]:6.2f}  {cm[1][1]:6.2f}]]   [FN%  TP%]]")
    print("="*60)
    
    # Check against targets (Requirement: Accuracy =68%, AUC =75%)
    print("\nQuality Targets:")
    acc_pass = "PASS" if metrics['accuracy'] >= 68.0 else "? FAIL"
    auc_pass = "PASS" if metrics['roc_auc'] >= 75.0 else "? FAIL"
    print(f"  Accuracy = 68%: {acc_pass} ({metrics['accuracy']:.2f}%)")
    print(f"  ROC-AUC  = 75%: {auc_pass} ({metrics['roc_auc']:.2f}%)")
    
    # Save outputs
    metrics_path = os.path.join(CONFIG["output_dir"], f"metrics_{CONFIG['model_type']}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # ROC curve
    plot_roc_curve(y_true, y_prob, os.path.join(CONFIG["output_dir"], "roc_curve.png"))
    
    # Confusion matrix
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix_raw']),
        os.path.join(CONFIG["output_dir"], "confusion_matrix.png")
    )
    
    # Misclassified examples
    save_misclassified_examples(
        y_true, y_pred, y_prob, ids,
        os.path.join(CONFIG["output_dir"], "misclassified.json"),
        top_n=10
    )
    
    print("\n? Evaluation complete!")
    return metrics

if __name__ == "__main__":
    main()