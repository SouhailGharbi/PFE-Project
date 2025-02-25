from transformers import AutoTokenizer, CamembertForSequenceClassification
import json
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

def load_dataset(file_path):
    """Charge le dataset JSON à partir du fichier donné."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier {file_path} introuvable.")
        return None
    except json.JSONDecodeError:
        print("❌ Erreur : Format JSON invalide.")
        return None

def extract_sql_identifiers(dataset):
    """Extrait les noms de tables et colonnes SQL du dataset."""
    identifiers = set()
    sql_pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")  # Détecte mots SQL

    sql_keywords = {
        "SELECT", "FROM", "WHERE", "AND", "OR", "IN", "LIKE", "JOIN",
        "ORDER", "GROUP", "BY", "HAVING", "COUNT", "SUM", "AVG", "MIN",
        "MAX", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
        "TABLE", "DATABASE"
    }
    for data in dataset:
        if 'output' in data:  # Vérifie que la clé 'output' est présente
            sql_query = data['output']
            words = sql_pattern.findall(sql_query)  # Extraction des mots
            normalized_query = " ".join([word.upper() if word.upper() in sql_keywords else word for word in words])
            data['output'] = normalized_query  # Mettre à jour la requête SQL normalisée

            # Filtrer pour éviter d'ajouter des mots-clés SQL comme tokens
            for word in words:
                if word.upper() not in sql_keywords:
                    identifiers.add(word)

    return list(identifiers)

def initialize_model_and_tokenizer(dataset):
    """Initialise le modèle et le tokenizer avec les tokens personnalisés."""
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    # Ajouter les tokens SQL
    sql_tokens = [
        "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "LIKE", "JOIN",
        "ORDER BY", "GROUP BY", "HAVING", "COUNT", "SUM", "AVG", "MIN", "MAX",
        "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TABLE", "DATABASE"
    ]

    # Ajouter les noms de tables et colonnes extraits du dataset
    table_column_tokens = extract_sql_identifiers(dataset)
    all_tokens = sql_tokens + table_column_tokens
    tokenizer.add_tokens(all_tokens)

    # Initialiser le modèle
    model = CamembertForSequenceClassification.from_pretrained("camembert-base")
    
    # Redimensionner les embeddings du modèle pour intégrer les nouveaux tokens
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenise les données et retourne une liste de tensors."""
    tokenized_dataset = []
    
    for i, data in enumerate(dataset):
        if 'input' not in data or 'output' not in data:
            print(f"⚠️ Donnée ignorée à l'index {i} : clé 'input' ou 'output' manquante.")
            continue
        
        input_text = f"Question: {data['input']} Réponse: {data['output']}"
        tokenized = tokenizer(
            input_text, 
            truncation=True, 
            padding='max_length',  # Assurez-vous que le padding est appliqué
            max_length=max_length,  # Limiter la longueur des séquences
            return_tensors="pt",
            add_special_tokens=True
        )
        
        tokenized_dataset.append({
            "input_text": input_text, 
            "input_ids": tokenized["input_ids"], 
            "attention_mask": tokenized["attention_mask"]
        })
        
        if i % 100 == 0:
            print(f"⏳ Tokenisation en cours... {i}/{len(dataset)}")

    print("✅ Tokenisation terminée.")
    return tokenized_dataset

class CustomDataset(Dataset):
    """Dataset personnalisé pour PyTorch."""
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        input_ids = self.tokenized_data[idx]['input_ids'].squeeze()
        attention_mask = self.tokenized_data[idx]['attention_mask'].squeeze()
        labels = self.tokenized_data[idx]['input_ids'].squeeze()

        # Vérification des dimensions avant de renvoyer les valeurs
        if input_ids.shape[0] == 0:
            print(f"⚠️ Warning: input_ids at index {idx} is empty!")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels  # Utilisation des ids comme labels pour l'entraînement
        }

def prepare_dataloader(tokenized_data, batch_size=8):
    """Prépare le DataLoader pour l'entraînement."""
    dataset = CustomDataset(tokenized_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, optimizer, device):
    """Entraîne le modèle."""
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Vérification des dimensions des tensors avant de passer à l'entraînement
        print(f"Input_ids shape: {input_ids.shape}")
        print(f"Attention_mask shape: {attention_masks.shape}")
        print(f"Labels shape: {labels.shape}")

        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"⏳ Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item()}")

def main():
    """Exécution principale du script."""
    dataset_path = "Train/Train_Data.json"
    dataset = load_dataset(dataset_path)
    
    if dataset is None:
        return

    # Initialiser le modèle et le tokenizer
    model, tokenizer = initialize_model_and_tokenizer(dataset)

    # Tokenisation des données
    tokenized_data = tokenize_dataset(dataset, tokenizer)

    # Préparer le DataLoader
    dataloader = prepare_dataloader(tokenized_data)

    # Définir l'optimiseur et l'appareil (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Entraînement du modèle
    print("🚀 Début de l'époque 1/3")
    train_model(model, dataloader, optimizer, device)

    print("✅ Entraînement terminé.")

if __name__ == "__main__":
    main()
