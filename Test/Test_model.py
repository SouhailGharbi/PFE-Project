from transformers import AutoTokenizer, CamembertForSequenceClassification
import torch

# 🏋️ Charger le modèle et le tokenizer entraînés
MODEL_PATH = "chemin/vers/votre_modele"  # 📌 Remplacez par le chemin réel
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = CamembertForSequenceClassification.from_pretrained(MODEL_PATH)

# Mettre le modèle en mode évaluation
model.eval()

def predict_question(question):
    """Fait une prédiction SQL à partir d'une question en langage naturel."""
    input_text = f"Question: {question}"
    
    # Tokenisation
    tokens = tokenizer(input_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # Prédiction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label

# 🔥 Mode interactif dans le terminal
print("\n💡 Tapez une question en langage naturel (ou 'exit' pour quitter) :")
while True:
    user_input = input("❓ Votre question : ")
    if user_input.lower() == "exit":
        print("👋 Fin du test.")
        break

    prediction = predict_question(user_input)
    print(f"🎯 Prédiction du modèle : {prediction}\n")
