#!/bin/bash
# Complete training pipeline for Conversational LLM
# Optimized for RTX 3050 4GB VRAM

set -e

echo "=============================================="
echo "CONVERSATIONAL LLM TRAINING PIPELINE"
echo "=============================================="

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found!"
    echo "Please run ./scripts/setup.sh first"
    exit 1
fi

# Check GPU
echo ""
echo "Checking GPU..."
python3 -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: CUDA not available! Training will be slow.')
else:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM: {memory:.1f} GB')
    if memory < 4:
        print('WARNING: Less than 4GB VRAM. May need to reduce batch size.')
"

# Step 1: Process data (if not already done)
if [ ! -f "data/processed/tokenized_data.pt" ]; then
    echo ""
    echo "=============================================="
    echo "[Step 1/3] Processing datasets..."
    echo "=============================================="
    python src/data_processing.py --base_dir . --max_seq_length 256 --vocab_size 8000 --max_samples 150000
else
    echo ""
    echo "[Step 1/3] Tokenized data already exists, skipping..."
fi

# Verify data
echo ""
echo "Verifying processed data..."
python3 -c "
import torch
data = torch.load('data/processed/tokenized_data.pt', weights_only=False)
print(f'  Sequences: {len(data[\"sequences\"]):,}')
print(f'  Vocab size: {data[\"vocab_size\"]:,}')
print(f'  Sequence length: {data.get(\"max_seq_length\", len(data[\"sequences\"][0]))}')
"

# Step 2: Train model
echo ""
echo "=============================================="
echo "[Step 2/3] Training model..."
echo "=============================================="
echo "This will take several hours. Progress will be shown below."
echo "Press Ctrl+C to stop training (progress is saved automatically)."
echo ""

python src/train.py --config configs/config.yaml

# Step 3: Run inference tests
echo ""
echo "=============================================="
echo "[Step 3/3] Running inference tests..."
echo "=============================================="

if [ -f "checkpoints/checkpoint_best.pt" ]; then
    CHECKPOINT="checkpoints/checkpoint_best.pt"
elif [ -f "checkpoints/checkpoint_latest.pt" ]; then
    CHECKPOINT="checkpoints/checkpoint_latest.pt"
else
    echo "No checkpoint found!"
    exit 1
fi

python src/inference.py \
    --checkpoint "$CHECKPOINT" \
    --tokenizer data/tokenizer/tokenizer.json \
    --mode test

echo ""
echo "=============================================="
echo "TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "To interact with your model:"
echo "  # Interactive chat:"
echo "  python src/inference.py --checkpoint $CHECKPOINT --mode chat"
echo ""
echo "  # Web interface (Gradio):"
echo "  python src/inference.py --checkpoint $CHECKPOINT --mode gradio"
echo ""
