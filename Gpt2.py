import pandas as pd
import json
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


Data = pd.read_json("conversational_pairs.json",lines=True)
print(type(Data))
pairs = []
for _, row in Data.iterrows():
        # Append tuples of (prompt, completion) to the pairs list
    pairs.append((row["prompt"], row["completion"]))

# Load the GPT-2 tokenizer and model
model_name = "gpt2"  # You can also use "gpt2-medium", "gpt2-large", or "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the padding token to be the same as the eos_token (End of Sentence token)
tokenizer.pad_token = tokenizer.eos_token

# Prepare the dataset for fine-tuning
def encode_data(examples):
    # Tokenize the input and output (prompt-completion pairs)
    # We use 'labels' as the same as 'input_ids' for language modeling tasks
    encodings = tokenizer(examples['prompt'], examples['completion'], padding="max_length", truncation=True, max_length=512)
    encodings['labels'] = encodings['input_ids']  # Set the labels to be the same as input_ids
    return encodings

# Convert to Hugging Face dataset
gpt_data = [{"prompt": pair[0], "completion": pair[1]} for pair in pairs]
gpt_dataset = Dataset.from_dict({
    "prompt": [item["prompt"] for item in gpt_data],
    "completion": [item["completion"] for item in gpt_data]
})

# Tokenize the dataset and add labels
tokenized_dataset = gpt_dataset.map(encode_data, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save results
    evaluation_strategy="epoch",  # Evaluation at the end of each epoch
    learning_rate=5e-5,  # Learning rate
    per_device_train_batch_size=2,  # Batch size per device
    num_train_epochs=3,  # Number of epochs
    save_steps=500,  # Save model every 500 steps
    logging_dir='./logs',  # Directory to store logs
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Using the same dataset for simplicity
)

# Start the training process
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_gpt2_model")

# Save the tokenizer for later use
tokenizer.save_pretrained("./fine_tuned_gpt2_tokenizer")

print("Fine-tuning complete and model saved.")
