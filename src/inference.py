"""
Inference and conversation testing for the trained model.
Includes interactive chat mode and evaluation utilities.
"""

import torch
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import create_model


class ConversationalInference:
    def __init__(self, checkpoint_path, tokenizer_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>"
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint or use defaults
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            config = checkpoint["config"]
            model_size = config.get("model_size", "xsmall")
            max_seq_length = config.get("max_seq_length", 256)
        else:
            model_size = "xsmall"
            max_seq_length = 256
        
        self.model = create_model(
            size=model_size,
            vocab_size=self.tokenizer.vocab_size,
            max_seq_length=max_seq_length
        ).to(self.device)
        
        # Load weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("Model loaded successfully!")
        print(f"Model size: {model_size}, Max sequence length: {max_seq_length}")
    
    @torch.no_grad()
    def generate(self, prompt, max_tokens=100, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text from a prompt"""
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def chat(self, user_message, conversation_history=""):
        """
        Generate a response in chat format.
        Maintains conversation context.
        """
        # Format the conversation
        if conversation_history:
            prompt = f"{conversation_history}\nHuman: {user_message}\nAssistant:"
        else:
            prompt = f"Human: {user_message}\nAssistant:"
        
        # Generate response
        full_output = self.generate(prompt, max_tokens=150, temperature=0.8)
        
        # Extract just the assistant's response
        if "Assistant:" in full_output:
            response = full_output.split("Assistant:")[-1].strip()
            # Stop at next speaker turn
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
        else:
            response = full_output[len(prompt):].strip()
        
        # Update conversation history
        new_history = f"{prompt} {response}"
        
        return response, new_history
    
    def interactive_chat(self):
        """Run interactive chat session"""
        print("\n" + "=" * 50)
        print("INTERACTIVE CHAT MODE")
        print("=" * 50)
        print("Commands:")
        print("  'quit' or 'exit' - Exit chat")
        print("  'reset' - Clear conversation history")
        print("  'story' - Generate a story")
        print("=" * 50 + "\n")
        
        history = ""
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "reset":
                history = ""
                print("[Conversation history cleared]\n")
                continue
            elif user_input.lower() == "story":
                print("\nGenerating a story...\n")
                story = self.generate("Once upon a time", max_tokens=200, temperature=0.9)
                print(f"Story: {story}\n")
                continue
            elif not user_input:
                continue
            
            response, history = self.chat(user_input, history)
            print(f"AI: {response}\n")
    
    def run_test_suite(self):
        """Run a suite of tests to evaluate conversational ability"""
        print("\n" + "=" * 50)
        print("RUNNING CONVERSATIONAL TEST SUITE")
        print("=" * 50)
        
        test_cases = [
            {
                "category": "Story Continuation",
                "tests": [
                    "Once upon a time, there was a little girl who",
                    "The dog ran into the park and",
                    "It was a sunny day, and the children decided to",
                    "One day, a little boy named Tom found a",
                ]
            },
            {
                "category": "Conversations",
                "tests": [
                    "Human: Hello!\nAssistant:",
                    "Human: How are you today?\nAssistant:",
                    "Human: What do you like to do?\nAssistant:",
                    "Human: Can you tell me a story?\nAssistant:",
                ]
            },
            {
                "category": "Sentence Completion",
                "tests": [
                    "The weather today is",
                    "I like to eat",
                    "My favorite color is",
                    "The little cat",
                ]
            },
        ]
        
        results = []
        
        for test_group in test_cases:
            print(f"\n--- {test_group['category']} ---")
            
            for test in test_group["tests"]:
                response = self.generate(test, max_tokens=80, temperature=0.8)
                print(f"\n  Prompt: {test}")
                print(f"  Output: {response[:150]}...")
                
                results.append({
                    "category": test_group["category"],
                    "prompt": test,
                    "response": response
                })
        
        print("\n" + "=" * 50)
        print("TEST SUITE COMPLETE")
        print("=" * 50)
        
        return results


def create_gradio_interface(inference_engine):
    """Create a Gradio web interface for testing"""
    import gradio as gr
    
    def chat_response(message, history):
        history_text = ""
        for h in history:
            history_text += f"Human: {h[0]}\nAssistant: {h[1]}\n"
        
        response, _ = inference_engine.chat(message, history_text)
        return response
    
    def generate_text(prompt, temperature, max_tokens):
        return inference_engine.generate(
            prompt,
            max_tokens=int(max_tokens),
            temperature=temperature
        )
    
    with gr.Blocks(title="Conversational LLM Tester", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Conversational LLM Testing Interface")
        gr.Markdown("Test your trained language model with various prompts and conversations.")
        
        with gr.Tab("💬 Chat"):
            chatbot = gr.ChatInterface(
                chat_response,
                title="Chat with the Model",
                description="Have a conversation with your trained model",
                examples=[
                    "Hello! How are you?",
                    "Tell me a short story",
                    "What's your favorite thing to do?",
                ]
            )
        
        with gr.Tab("✍️ Text Generation"):
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    temp_slider = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                    tokens_slider = gr.Slider(10, 300, value=100, label="Max Tokens")
                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output_text = gr.Textbox(label="Generated Text", lines=10)
            
            gr.Examples(
                examples=[
                    ["Once upon a time"],
                    ["The little dog ran to the"],
                    ["Human: Hello!\nAssistant:"],
                ],
                inputs=prompt_input
            )
            
            generate_btn.click(
                generate_text,
                inputs=[prompt_input, temp_slider, tokens_slider],
                outputs=output_text
            )
        
        with gr.Tab("🧪 Test Suite"):
            gr.Markdown("Run a comprehensive test suite to evaluate the model's capabilities.")
            run_tests_btn = gr.Button("Run Test Suite", variant="primary")
            test_output = gr.JSON(label="Test Results")
            
            run_tests_btn.click(
                lambda: inference_engine.run_test_suite(),
                outputs=test_output
            )
    
    return demo


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Conversational LLM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="./data/tokenizer/tokenizer.json")
    parser.add_argument("--mode", type=str, choices=["chat", "generate", "test", "gradio"], default="chat")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation mode")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio interface")
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = ConversationalInference(args.checkpoint, args.tokenizer)
    
    if args.mode == "chat":
        engine.interactive_chat()
    elif args.mode == "generate":
        if not args.prompt:
            args.prompt = input("Enter prompt: ")
        output = engine.generate(args.prompt)
        print(f"\nGenerated: {output}")
    elif args.mode == "test":
        engine.run_test_suite()
    elif args.mode == "gradio":
        demo = create_gradio_interface(engine)
        demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
