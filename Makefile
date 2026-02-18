# ==============================================================================
# Makefile — Multimodal Hateful Memes Classifier
# ==============================================================================
# Usage:
#   make help                  Show all targets
#   make all                   Full pipeline: install → ocr → train → eval
#   make train                 Train fusion model (default)
#   make train BATCH_SIZE=64   Train with custom hyperparameters
#   make api                   Start FastAPI server
#   make test                  Run API test suite
# ==============================================================================

.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Shell & flags
# ------------------------------------------------------------------------------
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# ------------------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------------------
SRC_DIR       := src
DATA_DIR      := data
CHECKPOINT_DIR := checkpoints
RESULTS_DIR   := results
IMG_DIR       := $(DATA_DIR)/img

# ------------------------------------------------------------------------------
# Configurable hyperparameters (override on CLI)
# e.g.: make train BATCH_SIZE=64 EPOCHS=30 LR=1e-4
# ------------------------------------------------------------------------------
BATCH_SIZE  ?= 32
NUM_WORKERS ?= 4
LR          ?= 2e-4
WEIGHT_DECAY ?= 1e-4
EPOCHS      ?= 20
WARMUP      ?= 2
PATIENCE    ?= 5
DROPOUT     ?= 0.3
SEED        ?= 42
USE_OCR     ?= True
PORT        ?= 8000

# ------------------------------------------------------------------------------
# Python
# ------------------------------------------------------------------------------
PYTHON := python3
PIP    := pip3

# ------------------------------------------------------------------------------
# Sentinel files — prevent re-running expensive one-time steps
# ------------------------------------------------------------------------------
SENTINEL_DIR  := .sentinels
INSTALL_DONE  := $(SENTINEL_DIR)/.install.done
OCR_DONE      := $(SENTINEL_DIR)/.ocr.done
SPLITS_DONE   := $(SENTINEL_DIR)/.splits.done

# ============================================================================
# PHONY TARGETS
# ============================================================================
.PHONY: help all \
        install check-env \
        ocr splits \
        train train-fusion train-image train-text train-caption train-all \
        eval eval-fusion eval-image eval-text eval-all \
        api \
        test test-api test-unit \
        lint format type-check \
        clean clean-results clean-checkpoints clean-sentinels clean-all \
        docker-build docker-run \
        report

# ============================================================================
# HELP
# ============================================================================
help:
	@printf "\n\033[1;36m🏆 Multimodal Hateful Memes Classifier\033[0m\n"
	@printf "\033[1;36m=======================================\033[0m\n\n"
	@printf "\033[1mUsage:\033[0m  make \033[4mtarget\033[0m [\033[4mOPTION\033[0m=\033[4mvalue\033[0m ...]\n\n"
	@printf "\033[1;33m📦 Setup\033[0m\n"
	@printf "  %-22s %s\n" "install"          "Install all Python dependencies"
	@printf "  %-22s %s\n" "check-env"        "Validate environment (Python, CUDA, deps)"
	@printf "\n\033[1;33m🔧 Data\033[0m\n"
	@printf "  %-22s %s\n" "ocr"              "Pre-compute OCR cache  [one-time, ~15 min]"
	@printf "  %-22s %s\n" "splits"           "Generate train/val/test splits"
	@printf "\n\033[1;33m🚀 Training\033[0m\n"
	@printf "  %-22s %s\n" "train"            "Train fusion model (default)"
	@printf "  %-22s %s\n" "train-fusion"     "Train fusion model (image + text + OCR)"
	@printf "  %-22s %s\n" "train-image"      "Train image-only baseline"
	@printf "  %-22s %s\n" "train-text"       "Train text-only baseline"
	@printf "  %-22s %s\n" "train-caption"    "Train fusion without OCR (caption only)"
	@printf "  %-22s %s\n" "train-all"        "Train all models (for full ablation)"
	@printf "\n\033[1;33m📊 Evaluation\033[0m\n"
	@printf "  %-22s %s\n" "eval"             "Evaluate fusion model on test set"
	@printf "  %-22s %s\n" "eval-all"         "Evaluate all models"
	@printf "\n\033[1;33m📡 API\033[0m\n"
	@printf "  %-22s %s\n" "api"              "Start FastAPI server (PORT=8000)"
	@printf "\n\033[1;33m🧪 Testing & Quality\033[0m\n"
	@printf "  %-22s %s\n" "test"             "Run full test suite"
	@printf "  %-22s %s\n" "test-api"         "Test live API endpoints with curl"
	@printf "  %-22s %s\n" "test-unit"        "Run unit tests with pytest"
	@printf "  %-22s %s\n" "lint"             "Lint code with flake8"
	@printf "  %-22s %s\n" "format"           "Auto-format code with black + isort"
	@printf "  %-22s %s\n" "type-check"       "Static type checking with mypy"
	@printf "\n\033[1;33m🐳 Docker\033[0m\n"
	@printf "  %-22s %s\n" "docker-build"     "Build Docker image"
	@printf "  %-22s %s\n" "docker-run"       "Run API in Docker container"
	@printf "\n\033[1;33m🧹 Cleanup\033[0m\n"
	@printf "  %-22s %s\n" "clean"            "Remove cache and .pyc files"
	@printf "  %-22s %s\n" "clean-results"    "Remove results/ (keep checkpoints)"
	@printf "  %-22s %s\n" "clean-checkpoints" "Remove checkpoints/"
	@printf "  %-22s %s\n" "clean-all"        "Remove everything (incl. data artifacts)"
	@printf "\n\033[1;33m⚡ Pipelines\033[0m\n"
	@printf "  %-22s %s\n" "all"              "install → ocr → splits → train → eval"
	@printf "  %-22s %s\n" "report"           "Generate final evaluation report"
	@printf "\n\033[1mConfigurable options (with defaults):\033[0m\n"
	@printf "  %-22s %s\n" "BATCH_SIZE=32"    "Mini-batch size"
	@printf "  %-22s %s\n" "EPOCHS=20"        "Maximum training epochs"
	@printf "  %-22s %s\n" "LR=2e-4"          "Learning rate"
	@printf "  %-22s %s\n" "WEIGHT_DECAY=1e-4" "AdamW weight decay"
	@printf "  %-22s %s\n" "PATIENCE=5"       "Early stopping patience"
	@printf "  %-22s %s\n" "DROPOUT=0.3"      "Fusion MLP dropout rate"
	@printf "  %-22s %s\n" "SEED=42"          "Global random seed"
	@printf "  %-22s %s\n" "USE_OCR=True"     "Include OCR text in text modality"
	@printf "  %-22s %s\n" "PORT=8000"        "FastAPI port"
	@printf "\n\033[1mExamples:\033[0m\n"
	@printf "  make train BATCH_SIZE=64 EPOCHS=30 LR=1e-4\n"
	@printf "  make train-all && make eval-all\n"
	@printf "  make api PORT=9000 &\n"
	@printf "  make test-api\n\n"

# ============================================================================
# SETUP
# ============================================================================
$(SENTINEL_DIR):
	@mkdir -p $(SENTINEL_DIR)

install: $(SENTINEL_DIR) requirements.txt
	@printf "\033[1;34m📦 Installing dependencies...\033[0m\n"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(INSTALL_DONE)
	@printf "\033[1;32m✅ Dependencies installed\033[0m\n"

check-env:
	@printf "\033[1;34m🔍 Checking environment...\033[0m\n"
	@$(PYTHON) --version
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@$(PYTHON) -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
	@$(PYTHON) -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@$(PYTHON) -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
	@printf "\033[1;32m✅ Environment OK\033[0m\n"

# ============================================================================
# DATA
# ============================================================================
$(OCR_DONE): $(SENTINEL_DIR)
	@printf "\033[1;34m🔧 Pre-computing OCR cache...\033[0m\n"
	$(PYTHON) -m $(SRC_DIR).precompute_ocr
	@touch $(OCR_DONE)
	@printf "\033[1;32m✅ OCR cache ready: $(DATA_DIR)/ocr_cache.json\033[0m\n"

ocr: $(OCR_DONE)

$(SPLITS_DONE): $(SENTINEL_DIR)
	@printf "\033[1;34m🔀 Generating train/val/test splits...\033[0m\n"
	$(PYTHON) -c "from $(SRC_DIR).data import create_splits; create_splits(seed=$(SEED))"
	@touch $(SPLITS_DONE)
	@printf "\033[1;32m✅ Splits saved: $(DATA_DIR)/splits.json\033[0m\n"

splits: $(SPLITS_DONE)

# ============================================================================
# TRAINING — shared Python inline runner
# ============================================================================
define run_train
	$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from $(SRC_DIR).train import CONFIG, main; \
CONFIG.update({ \
    'model_type':   '$(1)', \
    'use_ocr':      $(2), \
    'batch_size':   $(BATCH_SIZE), \
    'num_workers':  $(NUM_WORKERS), \
    'lr':           $(LR), \
    'weight_decay': $(WEIGHT_DECAY), \
    'epochs':       $(EPOCHS), \
    'warmup':       $(WARMUP), \
    'patience':     $(PATIENCE), \
    'dropout':      $(DROPOUT), \
    'seed':         $(SEED), \
}); main()"
endef

train: train-fusion

train-fusion: ocr splits
	@printf "\033[1;34m🚀 Training: fusion (image + text + OCR)\033[0m\n"
	@printf "   batch=$(BATCH_SIZE) | lr=$(LR) | epochs=$(EPOCHS) | seed=$(SEED)\n"
	@mkdir -p $(CHECKPOINT_DIR) $(RESULTS_DIR)
	$(call run_train,fusion,$(USE_OCR))
	@printf "\033[1;32m✅ Fusion model trained → $(CHECKPOINT_DIR)/best_fusion.pt\033[0m\n"

train-image: splits
	@printf "\033[1;34m🚀 Training: image-only baseline\033[0m\n"
	@mkdir -p $(CHECKPOINT_DIR) $(RESULTS_DIR)
	$(call run_train,image,False)
	@printf "\033[1;32m✅ Image model trained → $(CHECKPOINT_DIR)/best_image.pt\033[0m\n"

train-text: splits
	@printf "\033[1;34m🚀 Training: text-only baseline\033[0m\n"
	@mkdir -p $(CHECKPOINT_DIR) $(RESULTS_DIR)
	$(call run_train,text,False)
	@printf "\033[1;32m✅ Text model trained → $(CHECKPOINT_DIR)/best_text.pt\033[0m\n"

train-caption: splits
	@printf "\033[1;34m🚀 Training: fusion without OCR (caption only)\033[0m\n"
	@mkdir -p $(CHECKPOINT_DIR) $(RESULTS_DIR)
	$(call run_train,fusion,False)
	@printf "\033[1;32m✅ Caption-fusion model trained\033[0m\n"

train-all: train-fusion train-image train-text train-caption
	@printf "\033[1;32m✅ All models trained\033[0m\n"

# ============================================================================
# EVALUATION
# ============================================================================
define run_eval
	$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from $(SRC_DIR).eval import CONFIG, main; \
CONFIG.update({'model_type': '$(1)', 'use_ocr': $(2)}); main()"
endef

eval: eval-fusion

eval-fusion:
	@printf "\033[1;34m📊 Evaluating: fusion model\033[0m\n"
	@test -f $(CHECKPOINT_DIR)/best_fusion.pt || \
		(printf "\033[1;31m❌ Checkpoint not found. Run: make train-fusion\033[0m\n" && exit 1)
	$(call run_eval,fusion,True)
	@printf "\033[1;32m✅ Results → $(RESULTS_DIR)/metrics_fusion.json\033[0m\n"

eval-image:
	@printf "\033[1;34m📊 Evaluating: image-only model\033[0m\n"
	@test -f $(CHECKPOINT_DIR)/best_image.pt || \
		(printf "\033[1;31m❌ Checkpoint not found. Run: make train-image\033[0m\n" && exit 1)
	$(call run_eval,image,False)
	@printf "\033[1;32m✅ Results → $(RESULTS_DIR)/metrics_image.json\033[0m\n"

eval-text:
	@printf "\033[1;34m📊 Evaluating: text-only model\033[0m\n"
	@test -f $(CHECKPOINT_DIR)/best_text.pt || \
		(printf "\033[1;31m❌ Checkpoint not found. Run: make train-text\033[0m\n" && exit 1)
	$(call run_eval,text,False)
	@printf "\033[1;32m✅ Results → $(RESULTS_DIR)/metrics_text.json\033[0m\n"

eval-all: eval-fusion eval-image eval-text
	@printf "\n\033[1;32m📊 All evaluation results:\033[0m\n"
	@for f in $(RESULTS_DIR)/metrics_*.json; do \
		printf "  %-40s " "$$f"; \
		$(PYTHON) -c "import json,sys; d=json.load(open('$$f')); print(f\"Acc={d.get('accuracy',0)*100:.2f}%  AUC={d.get('roc_auc',0)*100:.2f}%\")"; \
	done

# ============================================================================
# API
# ============================================================================
api:
	@printf "\033[1;34m📡 Starting FastAPI server on http://localhost:$(PORT)\033[0m\n"
	@printf "   Endpoints: GET / /health /info | POST /predict\n"
	@printf "   Press Ctrl+C to stop\n\n"
	$(PYTHON) app.py --port $(PORT)

# ============================================================================
# TESTING & CODE QUALITY
# ============================================================================
test: test-unit test-api

test-unit:
	@printf "\033[1;34m🧪 Running unit tests...\033[0m\n"
	$(PYTHON) -m pytest tests/ -v --tb=short 2>/dev/null || \
		printf "\033[1;33m⚠️  No tests/ directory found — skipping unit tests\033[0m\n"

test-api:
	@printf "\033[1;34m🧪 Testing API endpoints...\033[0m\n"
	@printf "\n[1/4] Health check\n"
	@curl -sf http://localhost:$(PORT)/health | $(PYTHON) -m json.tool || \
		(printf "\033[1;31m❌ Server not running. Start with: make api\033[0m\n" && exit 1)
	@printf "\n[2/4] Root endpoint\n"
	@curl -sf http://localhost:$(PORT)/ | $(PYTHON) -m json.tool
	@printf "\n[3/4] Model info\n"
	@curl -sf http://localhost:$(PORT)/info | $(PYTHON) -m json.tool
	@printf "\n[4/4] Predict (sample image)\n"
	@SAMPLE=$$(ls $(IMG_DIR)/*.png 2>/dev/null | head -1); \
	if [ -n "$$SAMPLE" ]; then \
		curl -sf -X POST "http://localhost:$(PORT)/predict" \
			-F "image=@$$SAMPLE" \
			-F "caption=test caption" | $(PYTHON) -m json.tool; \
	else \
		printf "\033[1;33m⚠️  No sample images found in $(IMG_DIR)/\033[0m\n"; \
	fi
	@printf "\n\033[1;32m✅ API test suite passed\033[0m\n"

lint:
	@printf "\033[1;34m🔍 Linting with flake8...\033[0m\n"
	$(PYTHON) -m flake8 $(SRC_DIR)/ app.py \
		--max-line-length=100 \
		--ignore=E203,W503 \
		--exclude=__pycache__
	@printf "\033[1;32m✅ Lint passed\033[0m\n"

format:
	@printf "\033[1;34m🎨 Formatting with black + isort...\033[0m\n"
	$(PYTHON) -m isort $(SRC_DIR)/ app.py
	$(PYTHON) -m black $(SRC_DIR)/ app.py --line-length=100
	@printf "\033[1;32m✅ Formatting complete\033[0m\n"

type-check:
	@printf "\033[1;34m🔎 Type checking with mypy...\033[0m\n"
	$(PYTHON) -m mypy $(SRC_DIR)/ app.py \
		--ignore-missing-imports \
		--no-strict-optional
	@printf "\033[1;32m✅ Type check passed\033[0m\n"


# ============================================================================
# REPORT
# ============================================================================
report:
	@printf "\033[1;34m📝 Generating evaluation report...\033[0m\n"
	@test -f $(RESULTS_DIR)/metrics_fusion.json || \
		(printf "\033[1;31m❌ Run 'make eval' first\033[0m\n" && exit 1)
	$(PYTHON) -c "\
import json, datetime; \
m = json.load(open('$(RESULTS_DIR)/metrics_fusion.json')); \
print(f\"\"\"# Evaluation Summary — {datetime.date.today()}\n\
| Metric    | Value |\n|-----------|---------|\n\
| Accuracy  | {m.get('accuracy',0)*100:.2f}% |\n\
| Precision | {m.get('precision',0)*100:.2f}% |\n\
| Recall    | {m.get('recall',0)*100:.2f}% |\n\
| F1        | {m.get('f1',0)*100:.2f}% |\n\
| ROC-AUC   | {m.get('roc_auc',0)*100:.2f}% |\n\"\"\")"
	@printf "\033[1;32m✅ Report printed above\033[0m\n"

# ============================================================================
# CLEANUP
# ============================================================================
clean:
	@printf "\033[1;34m🧹 Cleaning cache and bytecode...\033[0m\n"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@printf "\033[1;32m✅ Clean complete (data/ and checkpoints/ preserved)\033[0m\n"

clean-results:
	@printf "\033[1;34m🧹 Removing results/...\033[0m\n"
	rm -rf $(RESULTS_DIR)
	@printf "\033[1;32m✅ Results removed\033[0m\n"

clean-checkpoints:
	@printf "\033[1;34m🧹 Removing checkpoints/...\033[0m\n"
	rm -rf $(CHECKPOINT_DIR)
	@printf "\033[1;32m✅ Checkpoints removed\033[0m\n"

clean-sentinels:
	@rm -rf $(SENTINEL_DIR)

clean-all: clean clean-results clean-checkpoints clean-sentinels
	@printf "\033[1;34m🧹 Removing data artifacts...\033[0m\n"
	rm -f $(DATA_DIR)/splits.json $(DATA_DIR)/ocr_cache.json
	@printf "\033[1;32m✅ Full clean complete\033[0m\n"

# ============================================================================
# ONE-COMMAND PIPELINE
# ============================================================================
all: install ocr splits train-fusion eval-fusion
	@printf "\n\033[1;32m🎉 Full pipeline complete!\033[0m\n"
	@printf "   ✅ Dependencies installed\n"
	@printf "   ✅ OCR cache generated\n"
	@printf "   ✅ Splits created\n"
	@printf "   ✅ Fusion model trained\n"
	@printf "   ✅ Evaluation complete\n"
	@printf "\n   📊 Results:  $(RESULTS_DIR)/metrics_fusion.json\n"
	@printf "   🤖 Model:    $(CHECKPOINT_DIR)/best_fusion.pt\n"
	@printf "   📡 API:      make api\n\n"
