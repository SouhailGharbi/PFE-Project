import json
import os
import torch
from transformers import AutoTokenizer, CamembertForSequenceClassification

# 📌 Charger les données d'entraînement
def load_sql_templates(train_data_path):
    """Charge le fichier JSON et crée un dictionnaire des requêtes SQL."""
    if not os.path.exists(train_data_path):
        print(f"❌ Erreur : Le fichier {train_data_path} est introuvable.")
        return {}

    try:
        with open(train_data_path, "r", encoding="utf-8") as file:
            train_data = json.load(file)
        return {i: entry["output"] for i, entry in enumerate(train_data)}
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de {train_data_path} : {e}")
        return {}

# 📌 Charger le modèle et le tokenizer
def load_model_and_tokenizer(model_path):
    """Charge le modèle et le tokenizer entraînés."""
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Le modèle '{model_path}' est introuvable.")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = CamembertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, None

# 📌 Tokenisation
def tokenize_question(question, tokenizer, max_length=128):
    """Tokenise la question et gère les erreurs éventuelles."""
    try:
        tokens = tokenizer(question, truncation=True, max_length=max_length, padding="longest", return_tensors="pt")
        return tokens["input_ids"], tokens["attention_mask"]
    except Exception as e:
        print(f"❌ Erreur de tokenisation : {e}")
        return None, None

# 📌 Prédiction
def predict(model, tokenizer, question):
    """Utilise le modèle pour prédire la classe correspondant à la question."""
    input_ids, attention_mask = tokenize_question(question, tokenizer)
    if input_ids is None or attention_mask is None:
        return None

    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        return torch.argmax(outputs.logits, dim=1).item()
    except Exception as e:
        print(f"❌ Erreur de prédiction : {e}")
        return None

# 📌 Générer la requête SQL
def generate_sql(prediction, sql_templates):
    """Retourne la requête SQL correspondant à la classe prédite."""
    return sql_templates.get(prediction, "⚠️ Aucune requête trouvée.") if prediction is not None else "⚠️ Erreur dans la prédiction."

# 📌 Fonction principale
def main():
    model_path = r"D:\PFE Project\Train\saved_model"
    train_data_path = r"D:\PFE Project\Train\Train_Data.json"

    sql_templates = load_sql_templates(train_data_path)
    if not sql_templates:
        print("❌ Impossible de charger les requêtes SQL. Vérifiez le fichier JSON.")
        return

    model, tokenizer = load_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        print("❌ Impossible de charger le modèle. Vérifiez le chemin.")
        return

    while True:
        question = input("💬 Entrez une question SQL (ou 'exit' pour quitter) : ")
        if question.lower() == "exit":
            break

        prediction = predict(model, tokenizer, question)
        sql_query = generate_sql(prediction, sql_templates)

        print("\n📝 Question SQL :", question)
        print("📌 Requête SQL générée :", sql_query, "\n")

# Exécuter le test
if __name__ == "__main__":
    main()
