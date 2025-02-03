import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import PreTrainedTokenizerFast, PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification  # Use if compatible
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Load ADE dataset
#dataset = load_dataset("clinc/clinc_oos")
dataset = load_dataset('ade-benchmark-corpus/ade_corpus_v2', 'Ade_corpus_v2_classification')

print(dataset)
# 2. Load custom SNOMEDTM model and tokenizer
#tokenizer = PreTrainedTokenizerFast.from_pretrained("./path/to/snomedtm-tokenizer")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


model = AutoModelForSequenceClassification.from_pretrained("./SNOMEDTM")


train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# 3. Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 4. Tokenize dataset
tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_datasets_test = test_dataset.map(tokenize_function, batched=True)
tokenized_datasets_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_dataset= tokenized_datasets_train
test_dataset = tokenized_datasets_test
#train_dataset = tokenized_datasets["train"]
#test_dataset = tokenized_datasets["test"]

# 5. Define evaluation metric
def compute_metrics1(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    
    # Compute accuracy
    acc = accuracy_score(labels, predictions)
    
    # Compute precision, recall, and F1-score for binary classification
    precision = precision_score(labels, predictions, pos_label=1)
    recall = recall_score(labels, predictions, pos_label=1)
    f1 = f1_score(labels, predictions, pos_label=1)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 6. Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",     # Set evaluation strategy to epoch
    save_strategy="epoch",           # Set save strategy to epoch
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,     # Works because save and eval strategies match
    metric_for_best_model="accuracy"
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 8. Fine-tune the model
trainer.train()

# 9. Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")
