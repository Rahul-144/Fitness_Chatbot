import pandas as pd
import json
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load the uploaded datasets
cleaned_data_path = "cleaned_data.csv"
exercise_data_path = "exercise_dataset.csv"

# Load datasets into pandas DataFrames
cleaned_data = pd.read_csv(cleaned_data_path)
exercise_data = pd.read_csv(exercise_data_path)

## Extract conversational pairs from cleaned_data.csv
def extract_cleaned_data_pairs(df):
    pairs = []
    for _, row in df.iterrows():
        product = row.get("product_title", "this product")
        if pd.notna(row.get("description")):
            pairs.append((f"What is {product}?", row["description"]))
        if pd.notna(row.get("directions")):
            pairs.append((f"How do I use {product}?", row["directions"]))
        if pd.notna(row.get("warning")):
            pairs.append((f"Are there any warnings for {product}?", row["warning"]))
        if pd.notna(row.get("goals")):
            pairs.append((f"What are the benefits of {product}?", row["goals"]))
    return pairs

# Extract conversational pairs from exercise_dataset.csv
def extract_exercise_data_pairs(df):
    pairs = []
    for _, row in df.iterrows():
        exercise = row["title"]
        pairs.append((f"What is {exercise}?", row["description"]))
        pairs.append((f"How do I perform {exercise}?", row["steps"]))
        pairs.append((f"What muscle groups does {exercise} target?", row["muscle_groups"]))
    return pairs

# Extract pairs from both datasets
cleaned_data_pairs = extract_cleaned_data_pairs(cleaned_data)
exercise_data_pairs = extract_exercise_data_pairs(exercise_data)

# Combine all pairs from both datasets
all_pairs = cleaned_data_pairs + exercise_data_pairs

# Convert the conversational pairs into GPT format (prompt-completion)
formatted_data = [{"prompt": f"User: {pair[0]}\nBot:", "completion": f" {pair[1]}"} for pair in all_pairs]

# Save the formatted data to a JSONL file
formatted_data_path = "Chatbot/conversational_pairs.json"
with open(formatted_data_path, "w") as f:
    for item in formatted_data:
        json.dump(item, f)
        f.write("\n")

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
gpt_data = [{"prompt": pair[0], "completion": pair[1]} for pair in all_pairs]
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
