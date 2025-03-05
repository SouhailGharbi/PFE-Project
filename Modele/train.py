import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
import json
import re
import spacy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Chargement du dataset ---
def load_dataset(file_path):
    """
    Fonction pour charger le dataset depuis un fichier JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)  # Charge le fichier JSON
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Erreur lors du chargement du fichier : {e}")
        return None  # Retourne None en cas d'erreur

# --- Extraction des identifiants SQL ---
def extract_sql_identifiers(dataset):
    """
    Fonction pour extraire les identifiants SQL (tables, colonnes) à partir des requêtes SQL.
    """
    identifiers = set()
    sql_pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")  # Expression régulière pour identifier les mots SQL
    sql_keywords = {"SELECT", "FROM", "WHERE", "AND", "OR", "IN", "LIKE", "JOIN",
                    "ORDER", "GROUP", "BY", "HAVING", "COUNT", "SUM", "AVG", "MIN",
                    "MAX", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
                    "TABLE", "DATABASE"}  # Liste des mots-clés SQL

    # Parcours des requêtes SQL dans le dataset
    for data in dataset:
        if 'output' in data:
            sql_query = data['output']
            words = sql_pattern.findall(sql_query)  # Trouver les mots dans la requête SQL
            # Normaliser la requête SQL en majuscules pour les mots-clés SQL
            normalized_query = " ".join([word.upper() if word.upper() in sql_keywords else word for word in words])
            data['output'] = normalized_query  # Met à jour la requête SQL dans le dataset
            # Ajoute les identifiants non mots-clés à la liste
            identifiers.update(word for word in words if word.upper() not in sql_keywords)
    
    return list(identifiers)  # Retourne la liste des identifiants SQL

# --- Initialisation du tokenizer ---
def initialize_tokenizer(dataset):
    """
    Fonction pour initialiser un tokenizer avec les mots français et les identifiants SQL.
    """
    tokenizer = AutoTokenizer.from_pretrained("t5-small")  # Chargement du tokenizer de base T5
    nlp = spacy.load("fr_core_news_sm")  # Chargement du modèle spaCy pour le traitement du texte en français

    # Récupération des mots français présents dans les entrées du dataset
    french_words = {token.text.lower() for data in dataset if 'input' in data for token in nlp(data['input']) if token.is_alpha}
    sql_keywords = ["SELECT", "FROM", "WHERE", "AND", "OR", "IN", "LIKE", "ORDER", "GROUP", "BY", "INNER", "JOIN","SUM", "*", "=", ">", "<", ">=", "<=", "!=", ";"]
    identifiers = extract_sql_identifiers(dataset)  # Extraire les identifiants SQL

    # Ajout des mots français, des mots-clés SQL et des identifiants à la liste des tokens du tokenizer
    tokenizer.add_tokens(list(french_words) + sql_keywords + identifiers, special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token  # Définir le token de padding à celui de fin de séquence
    return tokenizer

# --- Tokenisation du dataset ---
def tokenize_dataset(dataset, tokenizer, max_length=128):
    """
    Fonction pour tokeniser les entrées (inputs) et les sorties (requêtes SQL) du dataset.
    """
    input_ids_list, attention_mask_list, labels_list = [], [], []  # Listes pour stocker les résultats tokenisés

    # Tokenisation de chaque élément du dataset
    for data in tqdm(dataset, desc="🔄 Tokenisation des données"):
        if 'input' not in data or 'output' not in data:
            continue

        # Tokenisation des entrées et des sorties avec padding et troncature
        tokenized_input = tokenizer(data['input'], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        tokenized_sql = tokenizer(data['output'], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        
        # Nettoyage des tokens pour enlever ceux inutiles (</s> et <pad>)
        tokens_input_clean = [tok for tok in tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'].squeeze(0).tolist()) if tok not in ['</s>', '<pad>']]
        tokens_sql_clean = [tok for tok in tokenizer.convert_ids_to_tokens(tokenized_sql['input_ids'].squeeze(0).tolist()) if tok not in ['</s>', '<pad>']]
        
        # Affichage pour débogage
        print(f"🔹 Question : {data['input']}")
        print(f"🔹 Tokens (NL) : {tokens_input_clean}")
        print(f"🔹 Requête SQL : {data['output']}")
        print(f"🔹 Tokens (SQL) : {tokens_sql_clean}\n{'-'*50}")

        # Ajout des résultats tokenisés dans les listes
        input_ids_list.append(tokenized_input["input_ids"].squeeze(0))
        attention_mask_list.append(tokenized_input["attention_mask"].squeeze(0))
        labels_list.append(tokenized_sql["input_ids"].squeeze(0))

    # Retourne les résultats sous forme de tenseurs PyTorch
    return torch.stack(input_ids_list), torch.stack(attention_mask_list), torch.stack(labels_list)

# --- Dataset PyTorch ---
class CustomDataset(Dataset):
    """
    Classe Dataset personnalisée pour PyTorch.
    """
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)  # Nombre d'éléments dans le dataset

    def __getitem__(self, idx):
        return { 'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'labels': self.labels[idx] }

# --- Entraînement du modèle avec Early Stopping et Régularisation ---
def train_model(model, train_dataloader, val_dataloader, optimizer, device, epochs=5, patience=2, dropout_rate=0.1, weight_decay=0.01):
    """
    Fonction d'entraînement du modèle avec arrêt anticipé (early stopping) et régularisation.
    """
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")  # Initialiser la meilleure perte de validation
    epochs_no_improve = 0  # Compteur pour le early stopping
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # Scheduler pour ajuster le taux d'apprentissage

    # Ajout de la régularisation L2 via weight_decay dans l'optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=weight_decay)

    # Ajout de dropout dans le modèle
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate

    # Boucle d'entraînement pour chaque époque
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        print(f"📚 Début de l'Époch {epoch+1}")

        # Entraînement sur chaque batch
        for batch in tqdm(train_dataloader, desc=f"Entraînement - Époch {epoch+1}"):
            optimizer.zero_grad()  # Remise à zéro des gradients
            batch = {key: val.to(device) for key, val in batch.items()}  # Déplacement des données sur le GPU
            outputs = model(**batch)  # Passage des données dans le modèle
            loss = outputs.loss  # Calcul de la perte
            loss.backward()  # Calcul du gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip des gradients pour éviter l'explosion
            optimizer.step()  # Mise à jour des poids
            total_train_loss += loss.item()

        scheduler.step()  # Mise à jour du scheduler
        train_loss = total_train_loss / len(train_dataloader)  # Perte moyenne d'entraînement
        train_losses.append(train_loss)

        # Évaluation sur le set de validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_dataloader)  # Perte moyenne de validation
        val_losses.append(val_loss)

        print(f"📉 Perte entraînement : {train_loss:.4f}, Perte validation : {val_loss:.4f}")

        # Early Stopping : si aucune amélioration après "patience" époques, arrêter l'entraînement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset du compteur
        else:
            epochs_no_improve += 1  # Incrémentation si pas d'amélioration

        if epochs_no_improve >= patience:
            print(f"🛑 Arrêt anticipé à l'époque {epoch+1} (pas d'amélioration depuis {patience} époques)")
            break  # Arrêt de l'entraînement

    return model, train_losses, val_losses

# --- Visualisation des pertes ---
def plot_losses(train_losses, val_losses):
    """
    Fonction pour afficher les courbes de pertes d'entraînement et de validation.
    """
    plt.plot(train_losses, label="Perte d'entraînement")
    plt.plot(val_losses, label="Perte de validation")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.title("Courbe de Perte d'Entraînement et Validation")
    plt.legend()
    plt.show()

# --- Enregistrement du modèle ---
def save_model(model, tokenizer, model_path="Modele/trained_model"):
    """
    Fonction pour enregistrer le modèle et le tokenizer.
    """
    try:
        model.save_pretrained(model_path)  # Enregistre les poids du modèle
        tokenizer.save_pretrained(model_path)  # Enregistre le tokenizer
        print(f"✅ Modèle et tokenizer enregistrés avec succès dans '{model_path}'")
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement du modèle : {e}")

# --- Script principal ---
def main():
    """
    Fonction principale qui coordonne tout le processus.
    """
    # Chargement du dataset
    dataset = load_dataset("Modele/Data.json")
    if not dataset:
        return

    # Initialisation du tokenizer
    tokenizer = initialize_tokenizer(dataset)
    
    # Séparation du dataset en ensembles d'entraînement et de validation
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = map(lambda d: CustomDataset(*tokenize_dataset(d, tokenizer)), [train_data, val_data])
    train_dataloader, val_dataloader = map(lambda ds: DataLoader(ds, batch_size=8, shuffle=True), [train_dataset, val_dataset])

    # Sélection du périphérique (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialisation du modèle
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    model.resize_token_embeddings(len(tokenizer))  # Adapter la taille des embeddings du modèle au tokenizer

    # Initialisation de l'optimiseur avec régularisation L2
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Entraînement du modèle avec régularisation et early stopping
    model, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, optimizer, device, dropout_rate=0.1)
    
    # Visualisation des pertes
    plot_losses(train_losses, val_losses)

    # Enregistrement du modèle et tokenizer
    save_model(model, tokenizer, "trained_model")

if __name__ == "__main__":
    main()  