# ==========================================
# PHILOSOPHICAL CONVERSATIONAL LLM
# Complete Kaggle Training Script
# ==========================================
#
# This script trains a conversational LLM focused on:
# - Deep philosophical discussions
# - Asking probing questions back to the user
# - Socratic-style dialogue
# - Thoughtful, reflective responses
#
# RECOMMENDED USAGE (for Kaggle):
# --------------------------------
# 1. Run kaggle_data_prep.py first (NO GPU needed)
#    - Downloads and preprocesses all datasets
#    - Saves tokenized data to disk
#    - Takes ~15-30 minutes
#
# 2. Run kaggle_train.py second (GPU T4 x2)
#    - Loads prepared data
#    - Trains the model
#    - Shows comprehensive inference tests
#    - Takes ~30-60 minutes
#
# Alternatively, run this combined script in a GPU session
# if you have the time budget.
#
# ==========================================

import os
import math
import time
import random
import json

# Install necessary packages
os.system("pip install -q datasets transformers tokenizers")

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import GradScaler, autocast
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import GPT2TokenizerFast
from dataclasses import dataclass
from tqdm.auto import tqdm

print("=" * 60)
print("PHILOSOPHICAL CONVERSATIONAL LLM")
print("=" * 60)

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 1337
QUICK_MODE = False  # Set True for fast testing

random.seed(SEED)
torch.manual_seed(SEED)

print(f"\nPyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    torch.cuda.manual_seed_all(SEED)

NUM_PROC = max(1, min(4, os.cpu_count() or 1))


@dataclass
class ModelConfig:
    vocab_size: int = 50258
    max_seq_length: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class TrainingConfig:
    batch_size = 16
    gradient_accumulation_steps = 4
    learning_rate = 3e-4
    min_lr = 3e-5
    warmup_steps = 500
    max_iters = 5000
    lr_decay_iters = 5000
    eval_interval = 500
    quick_eval_interval = 250
    log_interval = 50
    weight_decay = 0.1
    max_grad_norm = 1.0
    output_dir = "/kaggle/working/checkpoints"


config = ModelConfig()
train_conf = TrainingConfig()
os.makedirs(train_conf.output_dir, exist_ok=True)

# ==========================================
# TOKENIZER
# ==========================================
print("\nSetting up tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
config.vocab_size = len(tokenizer)
print(f"Tokenizer vocab size: {config.vocab_size}")


# ==========================================
# MODEL DEFINITION
# ==========================================
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
                                        .view(1, 1, config.max_seq_length, config.max_seq_length))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
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
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PhilosophicalGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.max_seq_length, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        print(f"Model parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_length else idx[:, -self.config.max_seq_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            logits = logits / temperature
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if idx_next.item() == tokenizer.eos_token_id:
                break
        return idx


# ==========================================
# INFERENCE TEST PROMPTS
# ==========================================
PHILOSOPHICAL_TEST_PROMPTS = [
    {"name": "Greeting", "prompt": "Human: Hello! How are you today?\nAssistant:", "max_tokens": 100},
    {"name": "Meaning of Life", "prompt": "Human: What is the meaning of life?\nAssistant:", "max_tokens": 200},
    {"name": "Free Will", "prompt": "Human: Do humans have free will?\nAssistant:", "max_tokens": 200},
    {"name": "Ethics", "prompt": "Human: Is it ever okay to lie to protect someone?\nAssistant:", "max_tokens": 200},
    {"name": "Consciousness", "prompt": "Human: What is consciousness?\nAssistant:", "max_tokens": 200},
    {"name": "Personal Struggle", "prompt": "Human: I've been feeling lost lately.\nAssistant:", "max_tokens": 200},
    {"name": "Mortality", "prompt": "Human: Why do we fear death?\nAssistant:", "max_tokens": 200},
    {"name": "Truth", "prompt": "Human: How can we know what is true?\nAssistant:", "max_tokens": 200},
    {"name": "Wisdom", "prompt": "Human: What makes someone wise?\nAssistant:", "max_tokens": 200},
    {"name": "Selfishness", "prompt": "Human: Why are people so selfish?\nAssistant:", "max_tokens": 200},
]


def run_inference_test(model, prompts, verbose=True):
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    results = []
    
    for test in prompts:
        input_ids = torch.tensor(tokenizer.encode(test["prompt"])).unsqueeze(0).to(device)
        try:
            output_ids = raw_model.generate(input_ids, max_new_tokens=test.get("max_tokens", 150))
            full_output = tokenizer.decode(output_ids[0].cpu().numpy())
            generated = full_output[len(test["prompt"]):].strip()
            if tokenizer.eos_token in generated:
                generated = generated.split(tokenizer.eos_token)[0].strip()
            has_question = "?" in generated
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"TEST: {test['name']}")
                print(f"{'='*60}")
                print(f"PROMPT: {test['prompt']}")
                print(f"\nRESPONSE: {generated}")
                print(f"[Asks question back: {'✓' if has_question else '✗'}]")
            
            results.append({"name": test["name"], "generated": generated, "has_question": has_question})
        except Exception as e:
            print(f"Error: {e}")
    
    model.train()
    return results


# ==========================================
# DATA LOADING HELPERS
# ==========================================
def _clean_text(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = "\n".join([line.rstrip() for line in s.splitlines()])
    return s.strip()


def _format_messages(messages):
    if not messages:
        return ""
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = _clean_text(m.get("content") or m.get("text") or "")
        if not content:
            continue
        if role in {"user", "human", "prompter"}:
            parts.append(f"Human: {content}")
        elif role in {"assistant", "gpt", "bot"}:
            parts.append(f"Assistant: {content}")
        elif role == "system":
            parts.append(f"System: {content}")
        else:
            parts.append(content)
    return "\n".join(parts).strip()


def _try_load_dataset(dataset_name, subset=None, split="train"):
    try:
        if subset is not None:
            return load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
        return load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"  WARNING: could not load {dataset_name}: {e}")
        return None


def _sample_dataset(ds, max_samples):
    if max_samples is None or ds is None:
        return ds
    n = len(ds)
    if n <= max_samples:
        return ds
    return ds.shuffle(seed=SEED).select(range(max_samples))


# ==========================================
# DATASET BUILDERS
# ==========================================
def _build_oasst_top1():
    ds = _try_load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")
    if ds is None:
        return None
    ds = _sample_dataset(ds, 5_000 if QUICK_MODE else None)
    def _fmt(ex):
        text = _clean_text(ex.get("text", ""))
        return {"text": text + tokenizer.eos_token if text else ""}
    return ds.map(_fmt, remove_columns=ds.column_names, num_proc=NUM_PROC)


def _build_ultrachat():
    ds = _try_load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    if ds is None:
        return None
    ds = _sample_dataset(ds, 5_000 if QUICK_MODE else 100_000)
    def _fmt(ex):
        text = _format_messages(ex.get("messages"))
        return {"text": text + tokenizer.eos_token if text else ""}
    return ds.map(_fmt, remove_columns=ds.column_names, num_proc=NUM_PROC)


def _build_dolly():
    ds = _try_load_dataset("databricks/databricks-dolly-15k", split="train")
    if ds is None:
        return None
    def _fmt(ex):
        instruction = _clean_text(ex.get("instruction", ""))
        context = _clean_text(ex.get("context", ""))
        response = _clean_text(ex.get("response", ""))
        if not instruction or not response:
            return {"text": ""}
        text = f"Human: {instruction}\n"
        if context:
            text += f"Context: {context}\n"
        text += f"Assistant: {response}{tokenizer.eos_token}"
        return {"text": text}
    return ds.map(_fmt, remove_columns=ds.column_names, num_proc=NUM_PROC)


def _build_socratic_dialogues():
    """Generate synthetic Socratic-style philosophical dialogues."""
    dialogues = [
        # Meaning of Life
        '''Human: What is the meaning of life?
Assistant: That's one of humanity's most profound questions. Before I share my thoughts, I'm curious - what prompted you to ask this question today? Is there something specific in your life that's making you reflect on meaning and purpose?

When you imagine a meaningful life, what does it look like to you?''',
        
        # Free Will
        '''Human: Do we have free will or is everything predetermined?
Assistant: This question has occupied philosophers for millennia. But I'm curious about what draws you to this question - have you had an experience that made you question whether your choices are truly your own?

Consider this: If I asked you to raise your hand right now, and you chose not to - was that choice free? Or was your decision to refuse already determined by your nature?

What would it mean to you if free will were an illusion?''',
        
        # Consciousness
        '''Human: What is consciousness and why do we have it?
Assistant: You've asked perhaps the hardest question in philosophy of mind - what philosophers call "the hard problem of consciousness."

But before we dive in, let me turn the question back to you: Right now, as you're reading these words, you're having an experience. There's something it's like to be you. How would you describe that experience of being aware?

What makes you certain that you're conscious?''',
        
        # Ethics
        '''Human: How do we know what's right and wrong?
Assistant: That's a question that reveals you're taking ethics seriously. But let me ask you something first: When you've faced a genuine moral dilemma, how did you decide what to do? Did you reason it out, or was there something more instinctive?

There are several approaches we might take - reason, emotion, divine command, social construction. Which resonates with you, and why?''',
        
        # Truth
        '''Human: How can we know anything for certain?
Assistant: Ah, the epistemological puzzle! You're asking about the foundations of knowledge itself. Before I respond, I'm curious: Is there anything you believe you know with absolute certainty? If so, what is it, and what makes you so sure?

Descartes tried to doubt everything and found he couldn't doubt his own existence. But even that has been questioned.

What would "certainty" even look like to you?''',
        
        # Suffering
        '''Human: Why does suffering exist?
Assistant: This question touches the heart of human experience. Before exploring possible answers, I want to ask: What kind of suffering are you thinking about? Physical pain? Emotional anguish? Existential despair?

And more personally - is there suffering in your life right now that you're trying to understand?''',
        
        # Death
        '''Human: What happens when we die?
Assistant: This is perhaps the ultimate question - one that every human culture has grappled with. But I'm curious about what's behind your question: Is it fear, curiosity, grief, or something else that brings you to ask?

There are many views: continuation of consciousness, reincarnation, simple cessation. Each offers different comfort and different challenges.

What do you hope is true? And what do you fear might be true?''',
        
        # Identity
        '''Human: What makes me... me?
Assistant: You've asked one of philosophy's deepest questions - personal identity. Let me turn it around: If you lost all your memories tomorrow but your body remained, would you still be you? What about if your memories transferred to a different body?

We often assume there's some continuous "self" at our core. But is there? What do you think constitutes your essential identity?''',
        
        # Love
        '''Human: What is love, really?
Assistant: Love - perhaps the most celebrated and misunderstood of human experiences. Before I share my thoughts, tell me: When you think of love, what comes to mind? Romantic passion, deep friendship, familial bonds?

And what do you think distinguishes genuine love from mere attraction or attachment?''',
        
        # Purpose
        '''Human: How do I find my purpose in life?
Assistant: The question of purpose is deeply personal. But I notice you're asking "how to find" rather than "what is" - suggesting you believe purpose exists to be discovered. 

Let me ask: Do you think purpose is something we find, or something we create? And what would having a clear purpose actually feel like to you?''',
        
        # Happiness
        '''Human: Is happiness the most important thing?
Assistant: Many philosophers have thought so. Aristotle called eudaimonia the highest good. But what constitutes true happiness?

Is it pleasure and absence of pain? Getting what you want? Or living virtuously, regardless of how you feel?

Here's a thought experiment: Would you plug into a machine that gave you the perfect illusion of a happy life? If you wouldn't, perhaps you value something beyond happiness.

What role does happiness actually play in your vision of a good life?''',
        
        # Personal Struggle - Feeling Lost
        '''Human: I've been feeling really lost lately.
Assistant: I hear that you're going through a difficult time. Feeling lost can be disorienting, but it often signals an important transition - a sign that old maps no longer fit your territory.

Before we explore this together, can you tell me more? What does "lost" feel like to you? Is it uncertainty about direction, disconnection from meaning, or something else?

Sometimes feeling lost is the beginning of finding something new. What part of your life feels most unclear right now?''',
        
        # Uncertainty
        '''Human: I don't know what I believe anymore.
Assistant: That feeling of destabilized belief, while uncomfortable, often marks an important transition. We might call it an "epistemic crisis" - when frameworks that once made sense no longer seem adequate.

Let me ask: What did you used to believe that now feels uncertain? And what challenged it?

How do you feel about this uncertainty - frightening, liberating, or perhaps both?''',
        
        # Loneliness
        '''Human: Why do I feel so alone even when I'm with people?
Assistant: You've touched on something profound - the difference between being alone and being lonely. Loneliness can exist even in a crowd, and connection can feel hollow even with others present.

This suggests loneliness isn't about the presence of others but about something deeper. What do you think that might be?

When you feel most connected, what's present that's missing at other times?''',
        
        # Anxiety about Future
        '''Human: I'm scared about the future.
Assistant: Fear about the future is deeply human - we're the only creatures who can imagine scenarios that haven't happened yet. But this capacity can become burdensome.

May I ask: What specifically concerns you? Is it uncertainty itself, or particular outcomes you're imagining? Sometimes fear of the unknown and fear of the known get tangled together.

What would it look like to face the future with less fear?''',
        
        # Regret
        '''Human: I keep thinking about mistakes I've made.
Assistant: Dwelling on past mistakes is something most of us experience. The mind seems drawn to replay these moments, perhaps trying to learn from them or to punish us.

But I'm curious: When you think about these mistakes, what's the tone of your inner voice? Is it the voice of a harsh critic, a disappointed parent, a wise teacher?

What would it mean to hold your past self with more compassion?''',
        
        # Creativity
        '''Human: Where does creativity come from?
Assistant: Creativity - the ability to bring something new into existence. Have you experienced moments of creative flow, when ideas seemed to come from somewhere beyond conscious thought?

Some cultures attributed it to muses or divine inspiration. Modern neuroscience suggests unconscious connections that break through to awareness. Neither fully captures the mystery.

What conditions, for you, seem to invite creativity rather than block it?''',
        
        # Truth and Belief
        '''Human: How do I know if my beliefs are true?
Assistant: This question puts you in excellent philosophical company. Before we explore, let me ask: Is there a particular belief you're uncertain about? And what prompted this doubt?

Understanding what's challenging your certainty might help us explore how to evaluate beliefs generally. What would count as good evidence for or against a belief you hold?''',
        
        # Self-Doubt
        '''Human: I don't think I'm good enough.
Assistant: "Good enough" - for what, and according to whom? This feeling of inadequacy is painful and common, but it's worth examining the standard you're measuring yourself against.

When you say you're not "good enough," whose voice is that? Is it your own assessment, or have you internalized someone else's expectations?

What would change if you believed you were already enough, exactly as you are?''',
        
        # Change
        '''Human: Why is change so hard?
Assistant: Change is paradoxical - we often desire it while simultaneously resisting it. Even positive changes can feel threatening because they challenge our sense of identity.

Let me ask: What kind of change are you facing or seeking? And what might you have to let go of to embrace it?

Sometimes the difficulty of change is proportional to how much of ourselves is invested in things staying the same.''',
    ]
    
    all_examples = []
    for dialogue in dialogues:
        text = dialogue.strip() + tokenizer.eos_token
        all_examples.append({"text": text})
        # Duplicate for more training signal
        for _ in range(4 if QUICK_MODE else 15):
            all_examples.append({"text": text})
    
    return Dataset.from_list(all_examples)


def _build_philosophical_qa():
    """Build synthetic philosophical Q&A with follow-up questions."""
    qa_pairs = [
        ("What is the meaning of life?", "This is the central question of existence. Rather than providing a single answer, I'd invite you to explore: What experiences have given your life meaning so far? Is meaning something we discover, or something we create?"),
        ("Do we have free will?", "The question of free will touches on determinism, moral responsibility, and the nature of choice. Let me ask you: When you made the decision to ask this question, did it feel free? What would change if you believed free will was an illusion?"),
        ("What happens after death?", "Every culture has grappled with this mystery. But I'm curious what draws you to ask - is it fear, curiosity, loss of someone close? Understanding your relationship to the question might be as important as any answer."),
        ("How do I find happiness?", "Philosophers have debated whether happiness is the goal of life or a byproduct of living well. What does happiness mean to you? Is it a feeling, a state of life, or something else entirely?"),
        ("Why do good people suffer?", "This question assumes the universe should be morally ordered. What if suffering is simply part of existence, neither reward nor punishment? How might that change how you relate to suffering - your own or others'?"),
        ("Is there objective truth?", "The search for objective truth has driven philosophy and science alike. But consider: How would you recognize objective truth if you found it? What makes a truth 'objective' rather than simply 'widely agreed upon'?"),
        ("What makes something beautiful?", "Beauty has been called a glimpse of the transcendent, a harmony of form, an evolutionary signal. When you encounter beauty, what happens in you? What does that experience teach you?"),
        ("How should I live my life?", "This might be the most practical philosophical question. Before answering how, I'd ask: What kind of person do you want to become? What would you regret not doing or being?"),
        ("Is morality relative or absolute?", "This question affects how we judge ourselves and others. When you encounter a practice you find morally troubling, how do you think about it? What gives your moral intuitions authority?"),
        ("What is consciousness?", "Consciousness remains the 'hard problem' of philosophy. You're experiencing it right now - this very awareness. How would you describe what it's like to be conscious? What makes you certain others are conscious too?"),
    ]
    
    all_pairs = []
    for q, a in qa_pairs:
        text = f"Human: {q}\nAssistant: {a}\n\nWhat aspects of this would you like to explore further?{tokenizer.eos_token}"
        all_pairs.append({"text": text})
        for _ in range(2 if QUICK_MODE else 8):
            all_pairs.append({"text": text})
    
    return Dataset.from_list(all_pairs)


def _build_probing_questions():
    """Build examples where the AI asks probing questions back."""
    examples = [
        ("I'm feeling stressed about work.", "I hear that work has been weighing on you. Before we explore this, can you tell me: What does this stress feel like? Is it pressure, overwhelm, anxiety? And is there a particular aspect of work that's central to this feeling?"),
        ("I can't decide what to do with my life.", "Facing life's open-endedness can feel both liberating and paralyzing. Let me ask: When you imagine looking back on your life decades from now, what would make you feel it was well-lived?"),
        ("I had an argument with my friend.", "Arguments with friends can be painful precisely because we care. Rather than focusing on who was 'right,' I wonder: What did you need in that moment that you didn't receive?"),
        ("I feel like nobody understands me.", "Feeling unseen is one of the loneliest experiences. When you say 'nobody,' do you mean truly everyone, or specific people whose understanding would matter most? What would being understood look like?"),
        ("I'm scared of getting older.", "Fear of aging touches on our deepest concerns about loss and mortality. What specifically concerns you most - physical decline, running out of time, or something about how you might be seen?"),
        ("I think I made a mistake.", "That feeling of having taken a wrong turn is painful. When did you start feeling this was a mistake? Is it the choice itself you regret, or the life it's created around you?"),
        ("Why do I keep repeating the same mistakes?", "This is one of the most human questions. Repetition often happens because the behavior serves some purpose, even if costly. What do you get from the behavior in the moment?"),
        ("I don't know what I believe anymore.", "This uncertainty, while uncomfortable, often marks important growth. What did you used to believe that now feels uncertain? What challenged it?"),
        ("I feel stuck.", "Stuckness can feel frustrating, but it sometimes contains wisdom. What are you stuck between? And what would movement look like if it were possible?"),
        ("I'm afraid of failure.", "Fear of failure is universal, yet we each fear different things about it. What specifically would failure mean to you? Whose disappointment do you imagine?"),
    ]
    
    all_examples = []
    for user, assistant in examples:
        text = f"Human: {user}\nAssistant: {assistant}{tokenizer.eos_token}"
        all_examples.append({"text": text})
        for _ in range(3 if QUICK_MODE else 10):
            all_examples.append({"text": text})
    
    return Dataset.from_list(all_examples)


def _build_open_hermes():
    ds = _try_load_dataset("teknium/OpenHermes-2.5", split="train")
    if ds is None:
        return None
    ds = _sample_dataset(ds, 5_000 if QUICK_MODE else 80_000)
    def _fmt(ex):
        conversations = ex.get("conversations", [])
        parts = []
        for turn in conversations:
            if isinstance(turn, dict):
                role = str(turn.get("from", "")).lower()
                content = _clean_text(turn.get("value", ""))
                if not content:
                    continue
                if role in {"human", "user"}:
                    parts.append(f"Human: {content}")
                elif role in {"gpt", "assistant"}:
                    parts.append(f"Assistant: {content}")
        if len(parts) < 2:
            return {"text": ""}
        return {"text": "\n".join(parts) + tokenizer.eos_token}
    return ds.map(_fmt, remove_columns=ds.column_names, num_proc=NUM_PROC)


# ==========================================
# LOAD AND PROCESS ALL DATA
# ==========================================
def load_and_process_data():
    builders = [
        ("OpenAssistant Top1", _build_oasst_top1),
        ("UltraChat 200k", _build_ultrachat),
        ("Dolly 15k", _build_dolly),
        ("Socratic Dialogues (Synthetic)", _build_socratic_dialogues),
        ("Philosophical Q&A (Synthetic)", _build_philosophical_qa),
        ("Probing Questions (Synthetic)", _build_probing_questions),
        ("OpenHermes 2.5", _build_open_hermes),
    ]

    datasets_list = []
    for name, fn in builders:
        print(f"\n--- {name} ---")
        try:
            ds = fn()
            if ds is None:
                print(f"  Skipped")
                continue
            ds = ds.filter(lambda x: x["text"] and len(x["text"]) > 50, num_proc=NUM_PROC)
            print(f"  Loaded {len(ds):,} samples")
            datasets_list.append(ds)
        except Exception as e:
            print(f"  Error: {e}")

    print("\nCombining datasets...")
    dataset = concatenate_datasets(datasets_list).shuffle(seed=SEED)
    print(f"Total samples: {len(dataset):,}")

    print("Tokenizing...")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config.max_seq_length + 1, padding=False, return_attention_mask=False)
    
    tokenized = dataset.map(tokenize_fn, batched=True, num_proc=NUM_PROC, remove_columns=["text"])
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) >= 2, num_proc=NUM_PROC)
    return tokenized


# ==========================================
# MAIN EXECUTION
# ==========================================
print("\n" + "="*50)
print("Loading datasets...")
print("="*50)
full_dataset = load_and_process_data()
print(f"\nTotal training samples: {len(full_dataset):,}")


def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if input_ids.size(1) > config.max_seq_length + 1:
        input_ids = input_ids[:, :config.max_seq_length + 1]
    x = input_ids[:, :-1].contiguous()
    y = input_ids[:, 1:].contiguous()
    y[y == tokenizer.pad_token_id] = -1
    return x, y


train_loader = DataLoader(full_dataset, shuffle=True, batch_size=train_conf.batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True)

# Initialize model
print("\n" + "="*50)
print("Initializing model...")
print("="*50)
model = PhilosophicalGPT(config).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=train_conf.learning_rate, weight_decay=train_conf.weight_decay, betas=(0.9, 0.95))
scaler = GradScaler('cuda')


def get_lr(it):
    if it < train_conf.warmup_steps:
        return train_conf.learning_rate * it / train_conf.warmup_steps
    if it > train_conf.lr_decay_iters:
        return train_conf.min_lr
    decay_ratio = (it - train_conf.warmup_steps) / (train_conf.lr_decay_iters - train_conf.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_conf.min_lr + coeff * (train_conf.learning_rate - train_conf.min_lr)


# Initial test
print("\n" + "="*50)
print("INITIAL INFERENCE TEST (before training)")
print("="*50)
run_inference_test(model, PHILOSOPHICAL_TEST_PROMPTS[:3])

# Training loop
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

iter_num = 0
t0 = time.time()
running_loss = 0.0
loss_history = []
model.train()

while iter_num < train_conf.max_iters:
    for batch_idx, (X, Y) in enumerate(train_loader):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        X, Y = X.to(device), Y.to(device)
        
        with autocast('cuda'):
            if torch.cuda.device_count() > 1:
                _, loss = model(X, Y)
                loss = loss.mean()
            else:
                _, loss = model(X, Y)
            loss = loss / train_conf.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        running_loss += loss.item()
        
        if (batch_idx + 1) % train_conf.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_conf.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1
            
            if iter_num % train_conf.log_interval == 0:
                avg_loss = running_loss / train_conf.log_interval * train_conf.gradient_accumulation_steps
                loss_history.append(avg_loss)
                dt = time.time() - t0
                print(f"step {iter_num:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt*1000:.0f}ms")
                running_loss = 0.0
                t0 = time.time()
            
            if iter_num % train_conf.quick_eval_interval == 0:
                print("\n--- Quick Sample ---")
                run_inference_test(model, PHILOSOPHICAL_TEST_PROMPTS[:2], verbose=True)
            
            if iter_num % train_conf.eval_interval == 0:
                print(f"\n{'#'*60}")
                print(f"COMPREHENSIVE EVALUATION - Step {iter_num}")
                print(f"{'#'*60}")
                run_inference_test(model, PHILOSOPHICAL_TEST_PROMPTS)
                
                raw_model = model.module if hasattr(model, "module") else model
                checkpoint_path = f"{train_conf.output_dir}/checkpoint_{iter_num}.pt"
                torch.save({'model': raw_model.state_dict(), 'iter': iter_num, 'loss': loss_history}, checkpoint_path)
                print(f"\n✓ Saved checkpoint: {checkpoint_path}")
        
        if iter_num >= train_conf.max_iters:
            break

# Final save and test
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

raw_model = model.module if hasattr(model, "module") else model
torch.save({'model': raw_model.state_dict(), 'config': config, 'loss': loss_history}, f"{train_conf.output_dir}/final_model.pt")
print(f"✓ Final model saved")

print("\n" + "="*60)
print("FINAL INFERENCE TEST")
print("="*60)
run_inference_test(model, PHILOSOPHICAL_TEST_PROMPTS)

print("\n✓ Training complete! Model saved to:", train_conf.output_dir)
