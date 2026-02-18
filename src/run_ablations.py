
import os
import sys
import json
import subprocess
import shutil

def run_config(model_type, use_ocr, output_name):
    """Run training + eval with specific config, save results"""
    print(f"\n Running: {model_type} (OCR={use_ocr}) ? {output_name}")
    
    # Update train.py config temporarily
    with open('src/train.py', 'r') as f:
        train_content = f.read()
    
    # Replace config values
    train_content = train_content.replace(
        '"model_type": "fusion"', 
        f'"model_type": "{model_type}"'
    )
    train_content = train_content.replace(
        '"use_ocr": True', 
        f'"use_ocr": {use_ocr}'
    )
    
    with open('src/train.py', 'w') as f:
        f.write(train_content)
    
    # Run training
    subprocess.run([sys.executable, '-m', 'src.train'], check=True)
    
    # Run evaluation
    subprocess.run([sys.executable, '-m', 'src.eval'], check=True)
    
    # Save metrics with specific name
    src = f"results/metrics_{model_type}.json"
    dst = f"results/metrics_{output_name}.json"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Saved: {dst}")
    
    # Restore original config
    with open('src/train.py', 'r') as f:
        train_content = f.read()
    train_content = train_content.replace(
        f'"model_type": "{model_type}"', 
        '"model_type": "fusion"'
    )
    train_content = train_content.replace(
        f'"use_ocr": {use_ocr}', 
        '"use_ocr": True'
    )
    with open('src/train.py', 'w') as f:
        f.write(train_content)
    
    return dst

def load_metrics(filepath):
    """Load metrics JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def format_table(results):
    """Generate markdown table exactly as per requirements"""
    header = "| Model | Accuracy % | Precision % | Recall % | F1 % | ROC-AUC % |"
    separator = "|-------|------------|-------------|----------|------|-----------|"
    
    rows = []
    for name, metrics in results.items():
        row = f"| {name} | {metrics['accuracy']:.2f} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1']:.2f} | {metrics['roc_auc']:.2f} |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)

def format_confusion_matrix(cm):
    """Format confusion matrix as per requirements"""
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    return f"[[{tn:6.2f}  {fp:6.2f}]    [TN%  FP%]\n [{fn:6.2f}  {tp:6.2f}]]   [FN%  TP%]]"

def main():
    print("?? Running Ablation Studies (Requirement 4)")
    
    results = {}
    
    # 1. Image-only baseline (Requirement 4.a)
    metrics_path = run_config("image", False, "image_only")
    results["Image-only (CLIP)"] = load_metrics(metrics_path)
    
    # 2. Text-only baseline (Requirement 4.a)
    metrics_path = run_config("text", False, "text_only")
    results["Text-only (MiniLM)"] = load_metrics(metrics_path)
    
    # 3. Fusion with OCR (main model - should already exist)
    if os.path.exists("results/metrics_fusion.json"):
        results["Fusion (concat)"] = load_metrics("results/metrics_fusion.json")
    else:
        metrics_path = run_config("fusion", True, "fusion")
        results["Fusion (concat)"] = load_metrics(metrics_path)
    
    # 4. Caption-only ablation (Requirement 4.b)
    metrics_path = run_config("fusion", False, "caption_only")
    results["Fusion (caption-only)"] = load_metrics(metrics_path)
    
    # Generate final table
    print("\n" + "="*70)
    print("?? FINAL RESULTS TABLE (Test Set, in %)")
    print("="*70)
    print(format_table({k: v for k, v in results.items() if 'caption-only' not in k}))
    print("="*70)
    
    # Confusion Matrix for main fusion model
    cm = results["Fusion (concat)"]['confusion_matrix_raw']
    print("\n**Confusion Matrix (%; normalized by true class):**")
    print("```")
    print(format_confusion_matrix(cm))
    print("```")
    
    # Save to report.md
    with open('report.md', 'w') as f:
        f.write("# Evaluation Report: Multimodal Hateful Memes Classifier\n\n")
        f.write("## Dataset Summary\n")
        f.write("- Total labeled samples: 9,000\n")
        f.write("- Class balance: 63.3% not-hateful, 36.7% hateful\n")
        f.write("- Splits: Train 70%, Val 10%, Test 20% (stratified)\n\n")
        
        f.write("## Method\n")
        f.write("- Image encoder: CLIP ViT-B/32 (frozen)\n")
        f.write("- Text encoder: MiniLM-L6-v2 (frozen)\n")
        f.write("- Fusion: Late concatenation ? BN ? ReLU ? Dropout(0.3) ? Linear\n")
        f.write("- Loss: BCEWithLogitsLoss with pos_weight\n")
        f.write("- Optimizer: AdamW, cosine LR, AMP mixed precision\n\n")
        
        f.write("## Results Table (Test Set, in %)\n\n")
        f.write(format_table({k: v for k, v in results.items() if 'caption-only' not in k}))
        f.write("\n\n")
        
        f.write("## Confusion Matrix (%; normalized by true class)\n\n```python\n")
        f.write(format_confusion_matrix(cm))
        f.write("\n```\n\n")
        
        f.write("## Ablation: Caption-only vs Caption+OCR\n\n")
        cap_only = results.get("Fusion (caption-only)", {})
        fusion = results["Fusion (concat)"]
        if cap_only:
            f.write(f"| Text Input | Accuracy % | ROC-AUC % |\n")
            f.write(f"|------------|------------|-----------|\n")
            f.write(f"| Caption-only | {cap_only.get('accuracy', 'N/A'):.2f} | {cap_only.get('roc_auc', 'N/A'):.2f} |\n")
            f.write(f"| **Caption+OCR** | **{fusion['accuracy']:.2f}** | **{fusion['roc_auc']:.2f}** |\n")
        
        f.write("\n## Quality Targets\n\n")
        acc_pass = "?" if fusion['accuracy'] >= 68.0 else "?"
        auc_pass = "?" if fusion['roc_auc'] >= 75.0 else "?"
        f.write(f"- Accuracy = 68%: {acc_pass} ({fusion['accuracy']:.2f}%)\n")
        f.write(f"- ROC-AUC = 75%: {auc_pass} ({fusion['roc_auc']:.2f}%)\n")
    
    print(f"\n full report saved to report.md")
    print("Ablation studies complete!")

if __name__ == "__main__":
    main()