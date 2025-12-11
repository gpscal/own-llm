"""
GPT-style transformer model for conversational language modeling.
Based on nanoGPT architecture with modifications for dialogue.
Optimized for low VRAM GPUs (RTX 3050 4GB).
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the GPT model"""
    vocab_size: int = 32000
    max_seq_length: int = 256  # Reduced for 4GB VRAM
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    use_gradient_checkpointing: bool = False
    
    # Presets
    @classmethod
    def xsmall(cls):
        """~15M parameters - for GPUs with <4GB VRAM (RTX 3050 Laptop)"""
        return cls(n_layer=4, n_head=4, n_embd=256, max_seq_length=256, dropout=0.1)
    
    @classmethod
    def small(cls):
        """~25M parameters - optimized for RTX 3050 4GB"""
        return cls(n_layer=6, n_head=6, n_embd=384, max_seq_length=256)
    
    @classmethod
    def medium(cls):
        """~50M parameters - for GPUs with 6GB+ VRAM"""
        return cls(n_layer=8, n_head=8, n_embd=512, max_seq_length=256)
    
    @classmethod
    def large(cls):
        """~85M parameters - for GPUs with 8GB+ VRAM"""
        return cls(n_layer=10, n_head=10, n_embd=640, max_seq_length=512)


class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention support"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Check for Flash Attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # Causal mask for manual attention
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
                .view(1, 1, config.max_seq_length, config.max_seq_length)
            )
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if self.flash:
            # Use Flash Attention (more memory efficient)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ConversationalGPT(nn.Module):
    """GPT model for conversational language modeling"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_seq_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply gradient checkpointing if enabled
        if config.use_gradient_checkpointing:
            self.gradient_checkpointing_enable()
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {n_params/1e6:.2f}M parameters")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory"""
        self.config.use_gradient_checkpointing = True
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.max_seq_length, f"Sequence length {T} exceeds max {self.config.max_seq_length}"
        
        # Get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        if self.config.use_gradient_checkpointing and self.training:
            for block in self.transformer.h:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.transformer.h:
                x = block(x)
        
        x = self.transformer.ln_f(x)
        
        # Calculate logits and loss
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None):
        """Generate tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_length else idx[:, -self.config.max_seq_length:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if end token
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
        
        return idx


def create_model(size="small", vocab_size=32000, max_seq_length=256, use_gradient_checkpointing=False):
    """Factory function to create model"""
    config_map = {
        "xsmall": ModelConfig.xsmall,
        "small": ModelConfig.small,
        "medium": ModelConfig.medium,
        "large": ModelConfig.large,
    }
    
    config = config_map[size]()
    config.vocab_size = vocab_size
    config.max_seq_length = max_seq_length
    config.use_gradient_checkpointing = use_gradient_checkpointing
    
    return ConversationalGPT(config)


def estimate_memory_usage(config: ModelConfig, batch_size: int, dtype=torch.float16):
    """Estimate GPU memory usage for training"""
    # Model parameters
    n_params = (
        config.vocab_size * config.n_embd +  # token embeddings
        config.max_seq_length * config.n_embd +  # position embeddings
        config.n_layer * (
            4 * config.n_embd * config.n_embd +  # attention
            8 * config.n_embd * config.n_embd +  # MLP (4x expansion)
            4 * config.n_embd  # layer norms
        ) +
        config.vocab_size * config.n_embd  # lm_head (tied with embeddings)
    )
    
    bytes_per_param = 2 if dtype == torch.float16 else 4
    
    # Model size
    model_size_mb = (n_params * bytes_per_param) / (1024 * 1024)
    
    # Activations (rough estimate)
    activation_size_mb = (batch_size * config.max_seq_length * config.n_embd * 
                          config.n_layer * 2 * bytes_per_param) / (1024 * 1024)
    
    # Gradients
    gradient_size_mb = model_size_mb
    
    # Optimizer states (Adam: 2 states per param)
    optimizer_size_mb = model_size_mb * 2
    
    total_mb = model_size_mb + activation_size_mb + gradient_size_mb + optimizer_size_mb
    
    return {
        "model_mb": model_size_mb,
        "activations_mb": activation_size_mb,
        "gradients_mb": gradient_size_mb,
        "optimizer_mb": optimizer_size_mb,
        "total_mb": total_mb,
        "n_params": n_params
    }


if __name__ == "__main__":
    # Test model creation and memory estimation
    print("Testing model configurations for RTX 3050 4GB:\n")
    
    for size in ["small", "medium"]:
        config = ModelConfig.small() if size == "small" else ModelConfig.medium()
        print(f"=== {size.upper()} ===")
        
        for batch_size in [2, 4, 8]:
            mem = estimate_memory_usage(config, batch_size)
            print(f"  Batch {batch_size}: ~{mem['total_mb']:.0f}MB (Model: {mem['n_params']/1e6:.1f}M params)")
        print()
