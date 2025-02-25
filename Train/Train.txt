from transformers import AutoTokenizer, CamembertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import re

def load_dataset(file_path):
    """Charge le dataset JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"❌ Fichier introuvable : {file_path}")
        return None
    except json.JSONDecodeError:
        print("❌ Format JSON invalide.")
        return None

def initialize_tokenizer(dataset):
    """Initialise le tokenizer et ajoute les tokens SQL."""
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    sql_keywords = [
        "SELECT", "FROM", "WHERE", "AND", "OR", "IN", "LIKE", "INNER JOIN", "ORDER", "GROUP", "BY",
        "HAVING", "COUNT", "SUM", "AVG", "MIN", "MAX", "INSERT", "UPDATE", "DELETE", "CREATE",
        "ALTER", "DROP", "TABLE", "DATABASE", "*", "=", ">", "<", ">=", "<=", "!=", ";"
    ]

    tokenizer.add_tokens(sql_keywords, special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, len(sql_keywords)

def correct_sql_tokens(tokens):
    """Corrige les tokens SQL sans changer les noms de colonnes/tables."""
    corrected_tokens = []
    temp_token = ""

    sql_keywords = {
        "select", "from", "where", "and", "or", "in", "like", "inner join", "order", "group", "by",
        "having", "count", "sum", "avg", "min", "max", "insert", "update", "delete", "create",
        "alter", "drop", "table", "database"
    }

    for token in tokens:
        if token.startswith("▁"):  # Détecte les débuts de mots
            if temp_token:
                corrected_tokens.append(temp_token if temp_token.lower() not in sql_keywords else temp_token.upper())
            temp_token = token[1:]  # Enlever le préfixe '▁'
        else:
            temp_token += token  # Concatène les morceaux de mot

    if temp_token:
        corrected_tokens.append(temp_token if temp_token.lower() not in sql_keywords else temp_token.upper())

    # Séparation des mots SQL collés accidentellement
    final_tokens = []
    for tok in corrected_tokens:
        split_tok = re.findall(r'[A-Za-z_]+|\d+|[=<>!*;]+', tok)  # Sépare SQL et identifiants
        final_tokens.extend(split_tok)

    return final_tokens

def tokenize_dataset(dataset, tokenizer, max_length=128):
    """Tokenise les questions et réponses SQL, applique un padding."""
    input_ids_list, attention_mask_list, labels_list = [], [], []

    print("\n🔍 Affichage des tokens avant l'entraînement :\n")

    for i, data in enumerate(dataset):
        if 'input' not in data or 'output' not in data:
            print(f"⚠️ Donnée ignorée à l'index {i} : clé(s) manquante(s).")
            continue

        label = data.get('label', 0)  # Label par défaut

        input_text = f"Question: {data['input']} Réponse: {data['output']}"
        tokenized = tokenizer(input_text, truncation=True, max_length=max_length, padding="longest", return_tensors="pt")

        input_ids_list.append(tokenized["input_ids"].squeeze(0))
        attention_mask_list.append(tokenized["attention_mask"].squeeze(0))
        labels_list.append(label)

        # Affichage des tokens de la question
        question_tokens = tokenizer.tokenize(data['input'])
        question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
        print(f"📌 Question: {data['input']}")
        print(f"   Tokens: {question_tokens}")
        print(f"   IDs: {question_ids}\n")

        # Affichage des tokens de la réponse SQL corrigée
        raw_sql_tokens = tokenizer.tokenize(data['output'])
        corrected_sql_tokens = correct_sql_tokens(raw_sql_tokens)
        sql_ids = tokenizer.convert_tokens_to_ids(corrected_sql_tokens)
        print(f"📌 Réponse SQL corrigée: {' '.join(corrected_sql_tokens)}")
        print(f"   Tokens: {corrected_sql_tokens}")
        print(f"   IDs: {sql_ids}\n")

    if not input_ids_list:
        raise ValueError("❌ Aucune donnée valide trouvée !")

    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels_list)

    print(f"✅ {len(input_ids_list)} échantillons valides utilisés.")
    return input_ids_padded, attention_mask_padded, labels_tensor

class CustomDataset(Dataset):
    """Dataset PyTorch personnalisé."""
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def train_model(model, dataloader, optimizer, epochs=3):
    """Entraîne le modèle."""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"📉 Époque {epoch + 1}, Perte moyenne : {avg_loss:.4f}")

    print("✅ Entraînement terminé.")

def main():
    """Exécution du script."""
    dataset_path = "Train/Train_Data.json"
    dataset = load_dataset(dataset_path)
    
    if dataset is None:
        return

    tokenizer, added_tokens = initialize_tokenizer(dataset)
    input_ids, attention_mask, labels = tokenize_dataset(dataset, tokenizer)

    custom_dataset = CustomDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(custom_dataset, batch_size=8, shuffle=True)

    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)

    # 🚨 **Mise à jour des embeddings après l'ajout des nouveaux tokens**
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    train_model(model, dataloader, optimizer)

if __name__ == "__main__":
    main()  