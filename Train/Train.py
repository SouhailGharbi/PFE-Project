from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import json

# 1️⃣ Charger le dataset pré-traité
class SQLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = f"Question: {self.data[idx]['input']} Réponse: {self.data[idx]['output']}"
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0)
        }

# 2️⃣ Initialiser le tokenizer et le modèle GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 3️⃣ Charger le dataset
dataset = SQLDataset("Train/Train_Data.json", tokenizer)

# 4️⃣ Configurer l'entraînement
training_args = TrainingArguments(
    output_dir="./gpt2_sql_model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# 5️⃣ Lancer l'entraînement
trainer.train()

# 6️⃣ Sauvegarder le modèle fine-tuné
trainer.save_model("./gpt2_sql_finetuned")
tokenizer.save_pretrained("./gpt2_sql_finetuned")

print("✅ Entraînement terminé ! Modèle sauvegardé dans ./gpt2_sql_finetuned")
