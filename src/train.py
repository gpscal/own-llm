"""
Training script for conversational LLM.
Optimized for RTX 3050 4GB VRAM with:
- Mixed precision (AMP)
- Gradient accumulation
- Gradient checkpointing
- Memory-efficient attention
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import PreTrainedTokenizerFast
import yaml
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from model import create_model, ModelConfig


class TokenizedDataset(Dataset):
    """Dataset for pre-tokenized sequences"""
    def __init__(self, data_path):
        print(f"Loading tokenized data from {data_path}...")
        data = torch.load(data_path, weights_only=False)
        self.sequences = data["sequences"]
        self.vocab_size = data["vocab_size"]
        self.max_seq_length = data.get("max_seq_length", 256)
        print(f"Loaded {len(self.sequences):,} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        # Input is all but last token, target is all but first token
        return seq[:-1], seq[1:]


class Trainer:
    def __init__(self, config_path="configs/config.yaml"):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Load tokenizer
        tokenizer_path = self.config["tokenizer_path"]
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>"
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Create model
        use_grad_checkpoint = self.config.get("use_gradient_checkpointing", True)
        self.model = create_model(
            size=self.config["model_size"],
            vocab_size=self.tokenizer.vocab_size,
            max_seq_length=self.config["max_seq_length"],
            use_gradient_checkpointing=use_grad_checkpoint
        ).to(self.device)
        
        # Compile model for faster training (PyTorch 2.0+)
        if self.config.get("compile_model", False) and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        # Load dataset
        self.dataset = TokenizedDataset(self.config["data_path"])
        
        # Calculate effective batch size
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 8)
        effective_batch = self.config["batch_size"] * self.grad_accum_steps
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 2),
            pin_memory=True,
            drop_last=True
        )
        
        # Optimizer with weight decay (exclude embeddings and layernorms)
        param_groups = self._get_param_groups()
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config["learning_rate"],
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available()  # Faster fused optimizer
        )
        
        # Learning rate scheduler
        self.total_steps = len(self.dataloader) * self.config["epochs"] // self.grad_accum_steps
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = self.config.get("use_amp", True)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Logging
        self.log_interval = self.config.get("log_interval", 50)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best loss tracking
        self.best_loss = float("inf")
        self.global_step = 0
    
    def _get_param_groups(self):
        """Create parameter groups with/without weight decay"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases, embeddings, or layer norms
            if "bias" in name or "wte" in name or "wpe" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return [
            {"params": decay_params, "weight_decay": self.config.get("weight_decay", 0.1)},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine decay"""
        warmup_steps = self.config.get("warmup_steps", 500)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))  # Min 10% of LR
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch):
        """Train for one epoch - memory optimized"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    logits, loss = self.model(inputs, targets)
                    loss = loss / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
            else:
                logits, loss = self.model(inputs, targets)
                loss = loss / self.grad_accum_steps
                loss.backward()
            
            # Free memory immediately
            del logits
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                
                # Clear cache periodically to prevent fragmentation
                if self.global_step % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "step": self.global_step
                })
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        return avg_loss, epoch_time
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  New best model saved (loss: {loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % self.config.get("save_every", 5) == 0:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint["epoch"]
    
    @torch.no_grad()
    def generate_sample(self, prompts=None):
        """Generate samples to check training progress"""
        self.model.eval()
        
        if prompts is None:
            prompts = [
                "Hello, how are you",
                "Once upon a time",
                "Human: What is your name?\nAssistant:",
                "The little dog"
            ]
        
        print("\n" + "-" * 30 + " SAMPLES " + "-" * 30)
        
        for prompt in prompts:
            try:
                tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                output = self.model.generate(
                    tokens,
                    max_new_tokens=60,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                    eos_token_id=self.eos_token_id
                )
                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"\nPrompt: {prompt}")
                print(f"Output: {text[:200]}...")
            except Exception as e:
                print(f"Generation error: {e}")
        
        print("-" * 70 + "\n")
        self.model.train()
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Model size: {self.config['model_size']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.config['batch_size'] * self.grad_accum_steps}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Dataset size: {len(self.dataset):,} sequences")
        print(f"Total training steps: {self.total_steps:,}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient checkpointing: {self.config.get('use_gradient_checkpointing', True)}")
        print("=" * 60 + "\n")
        
        start_epoch = 0
        
        # Resume from checkpoint if exists
        resume_path = self.config.get("resume_from")
        if resume_path and os.path.exists(resume_path):
            start_epoch = self.load_checkpoint(resume_path) + 1
            print(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, self.config["epochs"]):
            avg_loss, epoch_time = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
            
            # Check for best model
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, avg_loss, is_best)
            
            # Generate samples
            if epoch % self.config.get("sample_every", 2) == 0:
                self.generate_sample()
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print(f"Best loss: {self.best_loss:.4f}")
        print("=" * 60)
        
        # Save final model (just weights)
        final_path = self.checkpoint_dir / "model_final.pt"
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved: {final_path}")


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    # Set memory optimization flags
    if torch.cuda.is_available():
        # Enable TF32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Memory optimization for fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Clear any existing cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Don't reserve all memory upfront
        torch.cuda.set_per_process_memory_fraction(0.90)
    
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
