import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image
import json
from config import *

def load_trained_model(model_path):
    """Load the fine-tuned model"""
    print(f"Loading trained model from {model_path}")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable native 2x faster inference
    FastVisionModel.for_inference(model)
    
    return model, tokenizer

def format_question_prompt(question, options, context_info=None):
    """Format a question for inference"""
    opts = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    
    ctx = []
    if context_info:
        if context_info.get("Indication"): 
            ctx.append(f"Indication: {context_info['Indication']}")
        if context_info.get("Comparison"): 
            ctx.append(f"Comparison: {context_info['Comparison']}")
        if context_info.get("Findings"): 
            ctx.append(f"Findings: {context_info['Findings']}")
        if context_info.get("Impression"): 
            ctx.append(f"Impression: {context_info['Impression']}")
    
    prompt = (
        "You are an expert radiologist. Use the chest X-ray image(s) and report context.\n"
        "Respond in this format:\n"
        f"  {REASONING_START}...{REASONING_END}{SOLUTION_START}LETTER - main answer{SOLUTION_END}\n\n"
        f"Question:\n{question}\n\nOptions:\n{opts}\n\nReport Context:\n" +
        ("\n".join(ctx) if ctx else "N/A")
    )
    
    return prompt

def run_inference(model, tokenizer, image_path, question, options, context_info=None):
    """Run inference on a single example"""
    
    # Load and prepare image
    image = Image.open(image_path).convert('RGB')
    
    # Format prompt
    prompt = format_question_prompt(question, options, context_info)
    
    # Create conversation format
    messages = [
        {
            "role": "user",
            "content": f"<image>\n{prompt}"
        }
    ]
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print("=" * 50)
    print("QUESTION:")
    print(question)
    print("\nOPTIONS:")
    for i, opt in enumerate(options):
        print(f"{chr(65+i)}. {opt}")
    print("\nMODEL RESPONSE:")
    print("-" * 30)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print("=" * 50)

def test_with_sample_data():
    """Test with sample data from the validation set"""
    
    # Load trained model
    model_path = TRAINING_CONFIG["output_dir"]  # Path to saved model
    model, tokenizer = load_trained_model(model_path)
    
    # Load a sample from validation data
    print("Loading sample from validation data...")
    
    try:
        with open(VAL_JSON, 'r') as f:
            val_data = json.load(f)
        
        # Get first sample
        sample_id = list(val_data.keys())[0]
        sample = val_data[sample_id]
        
        # Extract information
        question = sample.get('question', '')
        options = sample.get('options', [])
        image_paths = sample.get('ImagePath', [])
        
        if not image_paths:
            print("No image path found in sample")
            return
            
        image_path = image_paths[0]  # Use first image
        
        # Context information
        context_info = {
            'Indication': sample.get('Indication', ''),
            'Findings': sample.get('Findings', ''),
            'Impression': sample.get('Impression', ''),
            'Comparison': sample.get('Comparison', '')
        }
        
        print(f"Testing with sample ID: {sample_id}")
        print(f"Image path: {image_path}")
        
        # Run inference
        run_inference(model, tokenizer, image_path, question, options, context_info)
        
        # Show ground truth
        print("\nGROUND TRUTH:")
        print(f"Correct answer: {sample.get('correct_answer', 'N/A')}")
        print(f"Explanation: {sample.get('correct_answer_explanation', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing with sample data: {e}")
        print("You can manually test by calling run_inference() with your own data")

def interactive_test():
    """Interactive testing mode"""
    model_path = TRAINING_CONFIG["output_dir"]
    model, tokenizer = load_trained_model(model_path)
    
    print("Interactive testing mode. Enter 'quit' to exit.")
    
    while True:
        try:
            image_path = input("\nEnter image path: ").strip()
            if image_path.lower() == 'quit':
                break
                
            question = input("Enter question: ").strip()
            if question.lower() == 'quit':
                break
                
            print("Enter options (one per line, empty line to finish):")
            options = []
            while True:
                option = input(f"Option {chr(65+len(options))}: ").strip()
                if not option:
                    break
                options.append(option)
            
            if not options:
                print("No options provided, skipping...")
                continue
                
            run_inference(model, tokenizer, image_path, question, options)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_with_sample_data()