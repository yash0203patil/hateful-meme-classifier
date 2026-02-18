import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel, CLIPImageProcessor

# ============ CONFIG ============
IMG_EMB_DIM = 512      # CLIP ViT-B/32 output dim
TXT_EMB_DIM = 384      # MiniLM output dim
FUSION_DIM = 256
DROPOUT = 0.3
NUM_CLASSES = 1        # Binary: BCEWithLogitsLoss expects 1 output

# ============ IMAGE ENCODER ============
class ImageEncoder(nn.Module):
    """CLIP-based image encoder (frozen or fine-tuned)"""
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        # Load CLIP vision model
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Freeze/unfreeze parameters
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # Unfreeze last 2 layers for fine-tuning
            for param in self.encoder.vision_model.encoder.layers[-2:].parameters():
                param.requires_grad = True
        
        # Projection to common embedding space
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, IMG_EMB_DIM)
    
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        # Use [CLS] token (first position)
        img_emb = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        return self.projection(img_emb)  # [B, IMG_EMB_DIM]

# ============ TEXT ENCODER ============
class TextEncoder(nn.Module):
    """MiniLM-based text encoder"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", freeze=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # Unfreeze last layer for fine-tuning
            for param in self.encoder.encoder.layer[-1:].parameters():
                param.requires_grad = True
        
        # Projection to common embedding space
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, TXT_EMB_DIM)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling weighted by attention mask
        token_embeddings = outputs.last_hidden_state  # [B, seq_len, hidden]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        txt_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return self.projection(txt_emb)  # [B, TXT_EMB_DIM]

# ============ BASELINE MODELS ============
class ImageOnlyModel(nn.Module):
    """Image-only baseline (CLIP vision encoder)"""
    def __init__(self, freeze_encoder=True):
        super().__init__()
        self.img_encoder = ImageEncoder(freeze=freeze_encoder)
        self.head = nn.Sequential(
            nn.BatchNorm1d(IMG_EMB_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(IMG_EMB_DIM, NUM_CLASSES)
        )
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        img_emb = self.img_encoder(pixel_values)
        return self.head(img_emb)

class TextOnlyModel(nn.Module):
    """Text-only baseline (MiniLM text encoder)"""
    def __init__(self, freeze_encoder=True):
        super().__init__()
        self.txt_encoder = TextEncoder(freeze=freeze_encoder)
        self.head = nn.Sequential(
            nn.BatchNorm1d(TXT_EMB_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(TXT_EMB_DIM, NUM_CLASSES)
        )
    
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        txt_emb = self.txt_encoder(input_ids, attention_mask)
        return self.head(txt_emb)

# ============ FUSION MODEL (Required) ============
class FusionModel(nn.Module):
    """Late fusion: concat(img_emb, txt_emb) -> MLP -> logits"""
    def __init__(self, freeze_encoders=True):
        super().__init__()
        self.img_encoder = ImageEncoder(freeze=freeze_encoders)
        self.txt_encoder = TextEncoder(freeze=freeze_encoders)
        
        # Fusion MLP
        combined_dim = IMG_EMB_DIM + TXT_EMB_DIM
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, FUSION_DIM),
            nn.BatchNorm1d(FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_DIM, NUM_CLASSES)
        )
    
    def forward(self, pixel_values, input_ids, attention_mask):
        img_emb = self.img_encoder(pixel_values)      # [B, IMG_EMB_DIM]
        txt_emb = self.txt_encoder(input_ids, attention_mask)  # [B, TXT_EMB_DIM]
        combined = torch.cat([img_emb, txt_emb], dim=1)  # [B, IMG+TXT]
        return self.fusion(combined)

# ============ UTILS ============
def get_model(model_type="fusion", freeze_encoders=True, device="cuda"):
    """Factory function to instantiate models"""
    if model_type == "image":
        model = ImageOnlyModel(freeze_encoder=freeze_encoders)
    elif model_type == "text":
        model = TextOnlyModel(freeze_encoder=freeze_encoders)
    elif model_type == "fusion":
        model = FusionModel(freeze_encoders=freeze_encoders)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model.to(device)

def count_params(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)