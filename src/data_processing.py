"""
Data processing module for conversational LLM training.
Memory-efficient version - focuses on clean conversational data.
"""

import os
import re
import random
from pathlib import Path
from tqdm import tqdm
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


class DatasetProcessor:
    def __init__(self, base_dir=".", max_seq_length=256, max_samples_per_dataset=150000):
        self.base_dir = Path(base_dir)
        self.max_seq_length = max_seq_length
        self.max_samples = max_samples_per_dataset
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.tokenizer_dir = self.base_dir / "data" / "tokenizer"
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    def is_clean_text(self, text):
        """Check if text is clean conversational English"""
        if not text or len(text) < 50:
            return False
        
        # Reject texts with too many abbreviations (like A. B. C.)
        abbrev_pattern = r'\b[A-Z]\.\s*[A-Z]\.'
        if len(re.findall(abbrev_pattern, text)) > 3:
            return False
        
        # Reject texts with @-@ artifacts
        if '@-@' in text or '@.@' in text:
            return False
        
        # Reject texts with too many numbers/dates
        if len(re.findall(r'\d{4}', text)) > 5:  # Too many years
            return False
        
        # Reject texts that are mostly uppercase or have weird patterns
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if upper_ratio > 0.3:
            return False
        
        # Reject very technical/scientific text
        if any(term in text.lower() for term in ['µm', 'diameter', 'species', 'specimen']):
            return False
        
        return True
    
    def load_tinystories_streaming(self, output_file):
        """Load TinyStories - clean children's stories"""
        print("Loading TinyStories (streaming)...")
        count = 0
        
        txt_files = ["TinyStories-train.txt", "TinyStoriesV2-GPT4-train.txt"]
        
        with open(output_file, "w", encoding="utf-8") as out_f:
            for txt_file in txt_files:
                filepath = self.raw_dir / "TinyStories" / txt_file
                if not filepath.exists():
                    continue
                
                print(f"  Processing {txt_file}...")
                current_story = []
                
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in tqdm(f, desc=f"  {txt_file}"):
                        if "<|endoftext|>" in line:
                            story = " ".join(current_story).strip()
                            # Clean the story
                            story = re.sub(r'\s+', ' ', story)
                            
                            if self.is_clean_text(story):
                                out_f.write(story + "\n<|endoftext|>\n")
                                count += 1
                                if count >= self.max_samples:
                                    print(f"  Reached limit of {self.max_samples}")
                                    return count
                            current_story = []
                        else:
                            current_story.append(line.strip())
                
                # Process last story
                if current_story:
                    story = " ".join(current_story).strip()
                    if self.is_clean_text(story):
                        out_f.write(story + "\n<|endoftext|>\n")
                        count += 1
                
                if count >= self.max_samples:
                    break
        
        print(f"TinyStories: {count} clean stories saved")
        return count
    
    def load_dailydialog_streaming(self, output_file):
        """Load DailyDialog - everyday conversations"""
        print("Loading DailyDialog...")
        count = 0
        max_dialog = self.max_samples // 2
        
        try:
            from datasets import load_dataset
            
            with open(output_file, "a", encoding="utf-8") as out_f:
                dataset = load_dataset(
                    "li2017dailydialog/daily_dialog", 
                    trust_remote_code=True,
                    split="train"
                )
                
                for item in tqdm(dataset, desc="  Processing"):
                    dialog = item.get("dialog", [])
                    if dialog and len(dialog) >= 2:
                        formatted = self._format_conversation(dialog)
                        if self.is_clean_text(formatted):
                            out_f.write(formatted + "\n<|endoftext|>\n")
                            count += 1
                            if count >= max_dialog:
                                break
                
                del dataset
            
            print(f"DailyDialog: {count} conversations saved")
            
        except Exception as e:
            print(f"Could not load DailyDialog: {e}")
            # Add some sample conversations manually
            print("Adding sample conversations...")
            sample_convos = self._get_sample_conversations()
            with open(output_file, "a", encoding="utf-8") as out_f:
                for conv in sample_convos:
                    out_f.write(conv + "\n<|endoftext|>\n")
                    count += 1
            print(f"Added {len(sample_convos)} sample conversations")
        
        return count
    
    def _format_conversation(self, dialog_turns):
        """Format dialog turns into a conversation string"""
        formatted = ""
        for i, turn in enumerate(dialog_turns):
            speaker = "Human" if i % 2 == 0 else "Assistant"
            turn_clean = turn.strip()
            turn_clean = re.sub(r'\s+', ' ', turn_clean)
            formatted += f"{speaker}: {turn_clean}\n"
        return formatted.strip()
    
    def _get_sample_conversations(self):
        """Fallback sample conversations for training"""
        return [
            "Human: Hello! How are you today?\nAssistant: I'm doing great, thank you for asking! How about you?\nHuman: I'm good too. What do you like to do for fun?\nAssistant: I enjoy talking with people and helping them. What about you?",
            "Human: What is your name?\nAssistant: I'm a helpful assistant. You can call me whatever you like!\nHuman: Nice to meet you!\nAssistant: Nice to meet you too! How can I help you today?",
            "Human: Can you tell me a story?\nAssistant: Once upon a time, there was a little rabbit who loved to explore. One day, the rabbit found a beautiful garden full of carrots. The rabbit was so happy!\nHuman: That's a nice story!\nAssistant: Thank you! I'm glad you liked it.",
            "Human: What's the weather like?\nAssistant: I don't know the current weather, but I hope it's nice where you are! Do you like sunny days or rainy days?\nHuman: I like sunny days.\nAssistant: Sunny days are wonderful! They're perfect for going outside and having fun.",
            "Human: I'm feeling a bit sad today.\nAssistant: I'm sorry to hear that. It's okay to feel sad sometimes. Would you like to talk about it?\nHuman: I just miss my friend.\nAssistant: Missing friends is hard. Maybe you could call them or send them a message? Friends like to know we're thinking of them.",
        ] * 100  # Repeat to have more samples
    
    def create_combined_dataset(self):
        """Create combined dataset - TinyStories + DailyDialog only"""
        print("\n" + "=" * 50)
        print("CREATING CLEAN COMBINED DATASET")
        print("(TinyStories + DailyDialog - No Wikipedia)")
        print("=" * 50)
        
        combined_path = self.processed_dir / "combined_training_data.txt"
        
        if combined_path.exists():
            combined_path.unlink()
        
        total = 0
        total += self.load_tinystories_streaming(combined_path)
        total += self.load_dailydialog_streaming(combined_path)
        
        print(f"\nTotal samples: {total}")
        
        # Shuffle
        print("Shuffling dataset...")
        self._shuffle_file(combined_path)
        
        print(f"Saved to: {combined_path}")
        return total
    
    def _shuffle_file(self, filepath):
        """Shuffle samples in file"""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        samples = content.split("<|endoftext|>")
        samples = [s.strip() for s in samples if s.strip()]
        
        print(f"  Shuffling {len(samples)} samples...")
        random.shuffle(samples)
        
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample + "\n<|endoftext|>\n")
        
        del samples, content
    
    def train_tokenizer(self, vocab_size=8000):
        """Train tokenizer - smaller vocab for small model"""
        print("\n" + "=" * 50)
        print(f"TRAINING TOKENIZER (vocab_size={vocab_size})")
        print("=" * 50)
        
        combined_path = self.processed_dir / "combined_training_data.txt"
        
        if not combined_path.exists():
            raise FileNotFoundError("Run create_combined_dataset first")
        
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
            min_frequency=3
        )
        
        print("Training tokenizer...")
        tokenizer.train(files=[str(combined_path)], trainer=trainer)
        
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        tokenizer_path = self.tokenizer_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        
        print(f"Tokenizer saved to {tokenizer_path}")
        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        
        return tokenizer
    
    def tokenize_dataset(self, tokenizer_path=None):
        """Tokenize the combined dataset"""
        print("\n" + "=" * 50)
        print("TOKENIZING DATASET")
        print("=" * 50)
        
        from transformers import PreTrainedTokenizerFast
        
        if tokenizer_path is None:
            tokenizer_path = self.tokenizer_dir / "tokenizer.json"
        
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>"
        )
        
        combined_path = self.processed_dir / "combined_training_data.txt"
        
        all_sequences = []
        current_tokens = []
        stride = self.max_seq_length // 2
        
        print("Tokenizing...")
        
        with open(combined_path, "r", encoding="utf-8") as f:
            current_text = []
            for line in tqdm(f, desc="Processing"):
                if "<|endoftext|>" in line:
                    text = " ".join(current_text).strip()
                    if text:
                        tokens = tokenizer.encode(text + "<|endoftext|>")
                        current_tokens.extend(tokens)
                    current_text = []
                    
                    while len(current_tokens) >= self.max_seq_length:
                        seq = current_tokens[:self.max_seq_length]
                        all_sequences.append(seq)
                        current_tokens = current_tokens[stride:]
                else:
                    current_text.append(line.strip())
        
        # Handle remaining
        if current_text:
            text = " ".join(current_text).strip()
            if text:
                tokens = tokenizer.encode(text + "<|endoftext|>")
                current_tokens.extend(tokens)
        
        while len(current_tokens) >= self.max_seq_length:
            seq = current_tokens[:self.max_seq_length]
            all_sequences.append(seq)
            current_tokens = current_tokens[stride:]
        
        print(f"Total sequences: {len(all_sequences):,}")
        
        tokenized_path = self.processed_dir / "tokenized_data.pt"
        torch.save({
            "sequences": all_sequences,
            "vocab_size": tokenizer.vocab_size,
            "max_seq_length": self.max_seq_length
        }, tokenized_path)
        
        print(f"Saved to: {tokenized_path}")
        
        return all_sequences


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--max_samples", type=int, default=150000)
    args = parser.parse_args()
    
    processor = DatasetProcessor(
        base_dir=args.base_dir,
        max_seq_length=args.max_seq_length,
        max_samples_per_dataset=args.max_samples
    )
    
    processor.create_combined_dataset()
    processor.train_tokenizer(vocab_size=args.vocab_size)
    processor.tokenize_dataset()
    
    print("\n" + "=" * 50)
    print("DATA PROCESSING COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
