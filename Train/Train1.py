from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import torch
from preprocessing import load_dataset, initialize_tokenizer, tokenize_dataset
from datasets import Dataset

# Charger le dataset
dataset_path = "Train/Train_Data.json"
dataset = load_dataset(dataset_path)

if dataset is None:
    raise ValueError("Le dataset n'a pas pu être chargé !")

# Initialiser le tokenizer
tokenizer = initialize_tokenizer(dataset)

if tokenizer is None:
    raise ValueError("Le tokenizer n'a pas pu être initialisé !")

# Tokeniser le dataset
tokenized_data = tokenize_dataset(dataset, tokenizer)

if not isinstance(tokenized_data, Dataset):
    raise ValueError("Le dataset tokenisé n'est pas un objet Dataset valide !")

# Séparation en train et validation
split_dataset = tokenized_data.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Charger le modèle GPT-2 français (ex : camembertGPT-2)
model_name = "dbddv01/gpt2-french"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ajuster le vocabulaire du modèle
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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Lancer l'entraînement
trainer.train()

# Sauvegarde du modèle et du tokenizer
model.save_pretrained("models/sql_gpt2")
tokenizer.save_pretrained("models/sql_gpt2")

print("✅ Entraînement terminé ! Modèle sauvegardé.")
