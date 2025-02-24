from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from preprocessing import load_dataset, initialize_tokenizer, tokenize_dataset

# Charger le dataset
dataset_path = "Train/Train_Data.json"
dataset = load_dataset(dataset_path)

if dataset is None:
    raise ValueError("Le dataset n'a pas pu être chargé !")

# Initialiser le tokenizer
tokenizer = initialize_tokenizer(dataset)

# Tokeniser le dataset
tokenized_data = tokenize_dataset(dataset, tokenizer)

# Charger le modèle GPT-2 pour l'entraînement
model = AutoModelForCausalLM.from_pretrained("camembert-base")
model.resize_token_embeddings(len(tokenizer))

# Vérification de la présence du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Définition des arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./models/sql_gpt2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Initialiser Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data
)

# Lancer l'entraînement
trainer.train()

# Sauvegarde du modèle
model.save_pretrained("models/sql_gpt2")
tokenizer.save_pretrained("models/sql_gpt2")

print("✅ Entraînement terminé ! Modèle sauvegardé.")
