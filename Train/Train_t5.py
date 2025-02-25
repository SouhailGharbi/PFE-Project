import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json

# Fonction pour charger les données
def load_dataset(file_path):
    """Charge les données depuis un fichier JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        print(f"Erreur lors du chargement du dataset : {e}")
        return None

# Fonction pour extraire les identifiants SQL (noms de tables et de colonnes)
def extract_sql_identifiers(dataset):
    """Extrait les identifiants SQL tels que les noms de tables et de colonnes."""
    identifiers = set()
    for entry in dataset:
        for key in entry.keys():
            if isinstance(key, str):
                identifiers.add(key)
    return list(identifiers)

# Initialisation du tokenizer avec des tokens SQL
def initialize_tokenizer(dataset):
    """Initialise le tokenizer avec des tokens SQL."""
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    
    # Ajouter des tokens SQL supplémentaires
    sql_tokens = [
        "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "LIKE", "JOIN",
        "ORDER BY", "GROUP BY", "HAVING", "COUNT", "SUM", "AVG", "MIN", "MAX",
        "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TABLE", "DATABASE"
    ]

    # Ajouter les identifiants SQL extraits dynamiquement
    table_column_tokens = extract_sql_identifiers(dataset)
    all_tokens = sql_tokens + table_column_tokens
    tokenizer.add_tokens(all_tokens)

    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokens du tokenizer : {all_tokens}")  # Affichage des tokens ajoutés
    return tokenizer

# Tokenisation du dataset
def tokenize_dataset(dataset, tokenizer):
    """Tokenise les données du dataset avec le tokenizer."""
    tokenized_data = []
    for entry in dataset:
        sql_query = entry.get("sql_query", "")
        tokenized_data.append(tokenizer(sql_query, padding=True, truncation=True, max_length=512, return_tensors="pt"))
    return tokenized_data

# Classe custom Dataset pour PyTorch
class SQLDataset(Dataset):
    """Dataset pour entraîner le modèle sur des requêtes SQL."""
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: tensor.squeeze(0) for key, tensor in self.data[idx].items()}

# Fonction d'entraînement du modèle
def train_model(model, train_dataset, epochs=3, batch_size=8, learning_rate=5e-5):
    """Entraîne le modèle avec les données."""
    from torch.optim import AdamW
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})
    
    return model

# Fonction pour sauvegarder le modèle
def save_model(model, tokenizer, model_dir="model"):
    """Sauvegarde le modèle et le tokenizer."""
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Modèle sauvegardé dans le répertoire {model_dir}")

# Chargement et ajustement du modèle
def load_model_and_resize_embeddings(tokenizer):
    """Charge le modèle et ajuste la taille de ses embeddings pour inclure les nouveaux tokens."""
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    
    # Ajuster la taille des embeddings du modèle pour correspondre à la taille du vocabulaire du tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    return model

# Main
def main():
    """Fonction principale pour exécuter l'entraînement du modèle."""
    dataset_path = "Train/Train_Data.json"
    dataset = load_dataset(dataset_path)
    
    if dataset is None:
        return

    tokenizer = initialize_tokenizer(dataset)
    tokenized_data = tokenize_dataset(dataset, tokenizer)

    # Création du dataset
    train_dataset = SQLDataset(tokenized_data)
    
    # Chargement du modèle et ajustement des embeddings
    model = load_model_and_resize_embeddings(tokenizer)
    
    # Entraînement du modèle
    model = train_model(model, train_dataset)

    # Sauvegarde du modèle
    save_model(model, tokenizer)

if __name__ == "__main__":
    main()
