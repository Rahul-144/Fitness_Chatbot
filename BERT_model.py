import pandas as pd
import json
from datasets import Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments

# Load the uploaded datasets
data = pd.read_json("conversational_pairs.json",lines=True)
print(type(data))
pairs = []
for _, row in data.iterrows():
        # Append tuples of (prompt, completion) to the pairs list
    pairs.append((row["prompt"], row["completion"]))



# Convert the conversational pairs into a format suitable for BERT fine-tuning
qa_data = []
for pair in pairs:
    question = pair[0]
    answer = pair[1]
    
    # Calculate the start and end positions in the context
    answer_start = answer.find(answer)
    answer_end = answer_start + len(answer)
    
    qa_data.append({
        "context": answer, 
        "question": question, 
        "answers": {"text": [answer], "answer_start": [answer_start], "answer_end": [answer_end]}
    })

# Convert to a dictionary with column names as keys
qa_data_dict = {
    "context": [entry["context"] for entry in qa_data],
    "question": [entry["question"] for entry in qa_data],
    "answers": [entry["answers"] for entry in qa_data]
}

# Convert to Hugging Face dataset
qa_dataset = Dataset.from_dict(qa_data_dict)

# Load pre-trained model and fast tokenizer
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenize the data with offset mapping
def tokenize_function(examples):
    # Tokenizing the question and context
    tokenized_inputs = tokenizer(
        examples["question"], 
        examples["context"], 
        truncation=True, 
        padding="max_length", 
        max_length=512,
        return_offsets_mapping=True  # This gives us the start and end positions of each token
    )
    
    # We now need to find the start and end token positions using the offset_mapping
    start_positions = []
    end_positions = []
    
    for i in range(len(examples["context"])):
        # Get the start and end character positions of the answer in the context
        answer_start = examples["answers"][i]["answer_start"][0]
        answer_end = answer_start + len(examples["answers"][i]["text"][0])
        
        # Get the offset mapping for each example
        offsets = tokenized_inputs["offset_mapping"][i]
        
        start_token = None
        end_token = None
        
        # Find the start and end tokens that correspond to the character positions of the answer
        for idx, (start, end) in enumerate(offsets):
            if start <= answer_start < end:
                start_token = idx
            if start < answer_end <= end:
                end_token = idx
        
        if start_token is None or end_token is None:
            start_positions.append(tokenizer.pad_token_id)
            end_positions.append(tokenizer.pad_token_id)
        else:
            start_positions.append(start_token)
            end_positions.append(end_token)
    
    # Add the start and end token positions to the tokenized inputs
    tokenized_inputs["start_positions"] = start_positions
    tokenized_inputs["end_positions"] = end_positions
    return tokenized_inputs

tokenized_dataset = qa_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save results
    evaluation_strategy="epoch",  # Evaluation at the end of each epoch
    learning_rate=2e-5,  # Learning rate
    per_device_train_batch_size=8,  # Batch size per device
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

# Start the training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_bert_model")

# Save the tokenizer for later use
tokenizer.save_pretrained("./fine_tuned_bert_tokenizer")

# Optionally, save the formatted data (JSONL format for GPT fine-tuning or other models)


print("Fine-tuning complete and model saved.")
