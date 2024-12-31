from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def test_model(prompt):
    """
    Test the fine-tuned GPT-2 model with user input.
    """
    # Load the fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("/content/gpt_model")
    tokenizer = GPT2Tokenizer.from_pretrained("/content/gpt-tokenizer")
    
    # Set model to evaluation mode
    model.eval()
    
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("GPT-2 Model Testing Interface")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nEnter your question: ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
            
        try:
            # Generate and print response
            generated_text = test_model(user_input)
            print("\nModel Response:")
            print(generated_text)
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()