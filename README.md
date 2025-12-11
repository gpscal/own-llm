# Conversational LLM Training

Train a small conversational language model from scratch using TinyStories, DailyDialog, and WikiText datasets.

**Optimized for RTX 3050 4GB VRAM** (but works on any NVIDIA GPU).

## Quick Start

```bash
# 1. Setup environment
./scripts/setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Run full training pipeline
./scripts/run_training.sh
```

Or step by step:

```bash
# Setup
./scripts/setup.sh
source venv/bin/activate

# Process data
python src/data_processing.py

# Train
python src/train.py --config configs/config.yaml

# Test
python src/inference.py --checkpoint checkpoints/checkpoint_best.pt --mode chat
```

## Project Structure

```
own-llm/
├── data/
│   ├── raw/              # Your downloaded datasets (TinyStories, DailyDialog, WikiText)
│   ├── processed/        # Processed and tokenized data
│   └── tokenizer/        # Trained BPE tokenizer
├── src/
│   ├── data_processing.py   # Dataset processing
│   ├── model.py             # GPT model architecture
│   ├── train.py             # Training loop
│   └── inference.py         # Inference and testing
├── configs/
│   └── config.yaml          # Training configuration
├── checkpoints/             # Saved model checkpoints
├── scripts/
│   ├── setup.sh             # Environment setup
│   └── run_training.sh      # Full training pipeline
└── requirements.txt
```

## Hardware Requirements

| GPU VRAM | Recommended Settings |
|----------|---------------------|
| 4GB (RTX 3050) | `batch_size: 4`, `grad_accum: 16`, `model: small` |
| 6GB (RTX 2060) | `batch_size: 8`, `grad_accum: 8`, `model: small` |
| 8GB (RTX 3070) | `batch_size: 16`, `grad_accum: 4`, `model: medium` |
| 12GB+ | `batch_size: 32`, `grad_accum: 2`, `model: large` |

## Configuration

Edit `configs/config.yaml` to adjust training parameters:

```yaml
# Model size: "small" (~25M), "medium" (~50M), "large" (~85M)
model_size: "small"
max_seq_length: 256

# Training
batch_size: 4
gradient_accumulation_steps: 16
learning_rate: 3.0e-4
epochs: 15

# Memory optimizations (keep these true for 4GB VRAM)
use_amp: true
use_gradient_checkpointing: true
```

## Inference Modes

### Interactive Chat
```bash
python src/inference.py --checkpoint checkpoints/checkpoint_best.pt --mode chat
```

### Text Generation
```bash
python src/inference.py --checkpoint checkpoints/checkpoint_best.pt --mode generate --prompt "Once upon a time"
```

### Web Interface (Gradio)
```bash
python src/inference.py --checkpoint checkpoints/checkpoint_best.pt --mode gradio
```
Then open http://localhost:7860 in your browser.

### Test Suite
```bash
python src/inference.py --checkpoint checkpoints/checkpoint_best.pt --mode test
```

## Training Time Estimates

For RTX 3050 4GB with default settings (~25M parameter model):

| Dataset Size | Training Time |
|--------------|---------------|
| 100K sequences | ~2-3 hours |
| 500K sequences | ~10-15 hours |
| 1M sequences | ~20-30 hours |

## Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce `batch_size` to 2
2. Increase `gradient_accumulation_steps` to 32
3. Ensure `use_amp: true` and `use_gradient_checkpointing: true`
4. Reduce `max_seq_length` to 128
5. Close other applications using the GPU

### Poor Generation Quality
1. Train for more epochs
2. Check that data processing completed successfully
3. Adjust `temperature` during generation (0.7-0.9 is usually good)
4. Use `top_k` and `top_p` sampling

### Training is Slow
1. Enable `compile_model: true` (PyTorch 2.0+)
2. Increase `batch_size` if memory allows
3. Use `num_workers: 4` for data loading

## Expected Results

After training on TinyStories + DailyDialog + WikiText, your model should:

- ✅ Generate fluent sentences with proper grammar
- ✅ Continue stories naturally
- ✅ Respond to greetings appropriately
- ✅ Maintain basic conversation context
- ✅ Use simple vocabulary correctly

### Example Outputs

```
Prompt: "Once upon a time"
Output: "Once upon a time, there was a little girl named Lily. 
        She loved to play in the park with her dog Max..."

Prompt: "Human: Hello!\nAssistant:"
Output: "Hello! How are you today? It's nice to meet you!"
```

## License

This project is for educational purposes. The model architecture is based on nanoGPT.
