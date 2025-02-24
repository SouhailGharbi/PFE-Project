from transformers import AutoTokenizer
import json
import re

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
    sql_pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")  # Détecte mots SQL # Expression régulière pour capturer les noms de colonnes/tables

    sql_keywords = {
        "SELECT", "FROM", "WHERE", "AND", "OR", "IN", "LIKE", "JOIN",
        "ORDER", "GROUP", "BY", "HAVING", "COUNT", "SUM", "AVG", "MIN",
        "MAX", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
        "TABLE", "DATABASE"
    }
    for data in dataset:
        if 'output' in data: # Vérifie que la clé 'output' est présente
            sql_query = data['output']
            words = sql_pattern.findall(sql_query) # Extraction des mots dans la requête SQL
            normalized_query = " ".join([word.upper() if word.upper() in sql_keywords else word for word in words])
            data['output'] = normalized_query  # Mettre à jour la requête SQL normalisée

            # Filtrer pour éviter d'ajouter des mots-clés SQL comme tokens
            for word in words:
                if word.upper() not in ["SELECT", "FROM", "WHERE", "AND", "OR", "IN", "LIKE", "JOIN",
                                        "ORDER", "GROUP", "BY", "HAVING", "COUNT", "SUM", "AVG", "MIN",
                                        "MAX", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
                                        "TABLE", "DATABASE"]:
                    identifiers.add(word)

    return list(identifiers)

def initialize_tokenizer(dataset):
    """Initialise le tokenizer CamemBERT et ajoute les tokens SQL + colonnes/tables."""
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    # Ajouter les mots-clés SQL
    sql_tokens = [
        "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "LIKE", "JOIN",
        "ORDER BY", "GROUP BY", "HAVING", "COUNT", "SUM", "AVG", "MIN", "MAX",
        "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TABLE", "DATABASE"
    ]

    # Ajouter les noms de tables et colonnes trouvés dynamiquement
    table_column_tokens = extract_sql_identifiers(dataset)
    all_tokens = sql_tokens + table_column_tokens
    tokenizer.add_tokens(all_tokens)

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
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
            padding=True,  
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

def main():
    """Exécution principale du script."""
    dataset_path = "Train/Train_Data.json"
    dataset = load_dataset(dataset_path)
    
    if dataset is None:
        return

    tokenizer = initialize_tokenizer(dataset)
    tokenized_data = tokenize_dataset(dataset, tokenizer)

    # Affichage du premier élément tokenisé
    if tokenized_data:
        print("\n🔍 Exemple de donnée tokenisée :")
        print(f"Texte d'origine : {tokenized_data[0]['input_text']}")
        print(f"Tokens : {tokenizer.convert_ids_to_tokens(tokenized_data[0]['input_ids'][0])}")
        print(f"IDs : {tokenized_data[0]['input_ids']}")
        print(f"Masque d'attention : {tokenized_data[0]['attention_mask']}")

if __name__ == "__main__":
    main()
