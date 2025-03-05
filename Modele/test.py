import torch 
from transformers import AutoTokenizer, T5ForConditionalGeneration
import spacy

# --- Chargement du modèle et du tokenizer ---
def load_model_and_tokenizer(model_path):
    """Charge le modèle et le tokenizer depuis le dossier sauvegardé."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou du tokenizer: {e}")
        return None, None

# --- Prétraitement de l'input avec spaCy ---
def preprocess_input_with_spacy(input_text):
    """Utilise spaCy pour analyser et traiter l'input en français."""
    nlp = spacy.load("fr_core_news_sm")  # Charger le modèle spaCy pour le français
    doc = nlp(input_text)  # Analyser le texte
    
    # Exemple de prétraitement : Lemmatization et extraction de tokens (vous pouvez ajouter plus de traitements ici)
    lemmatized_tokens = [token.lemma_ for token in doc if token.is_alpha]  # Lemmatization et filtrage des tokens alpha
    preprocessed_input = " ".join(lemmatized_tokens)
    
    # Affichage pour débogage
    print("Texte prétraité avec spaCy :")
    print(preprocessed_input)
    
    return preprocessed_input

# --- Génération de la requête SQL ---
def generate_sql(prompt, tokenizer, model, device="cpu", max_length=128):
    """Génère une requête SQL à partir d'un prompt en français."""
    # Prétraitement de l'input avec spaCy
    preprocessed_prompt = preprocess_input_with_spacy(prompt)
    
    # Formater le prompt pour le modèle (ajout des balises [NL] et [SQL])
    input_text = f"[NL] {preprocessed_prompt} [SQL]"

    # Tokenisation
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Envoi sur le bon périphérique (CPU/GPU)
    
    # Affichage des tokens pour débogage (peut être supprimé une fois testé)
    print("Tokens d'entrée :", inputs["input_ids"])
    print("Tokens décodés :", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
    
    # Génération de la requête SQL
    with torch.no_grad():  # Désactive le calcul du gradient pour l'inférence
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,  # Utiliser la recherche en faisceau
            early_stopping=True,  # Arrêter la génération si le modèle termine prématurément
            no_repeat_ngram_size=2,  # Éviter les répétitions de n-grammes
        )
    
    # Détokenisation
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

# --- Script principal ---
def main():
    # Chemin vers le modèle sauvegardé
    model_path = "saved_model"
    
    # Chargement du modèle et du tokenizer
    tokenizer, model = load_model_and_tokenizer(model_path)
    if tokenizer is None or model is None:
        return  # Si le modèle ou le tokenizer n'ont pas pu être chargés, on arrête

    # Configuration du périphérique (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prompt de test (question en français)
    prompt = "Quels employés travaillent dans le département 10 et gagnent plus de 5000€ ?"
    
    # Génération de la requête SQL
    sql_query = generate_sql(prompt, tokenizer, model, device)
    
    # Affichage des résultats
    print(f"🔹 Prompt (Français) : {prompt}")
    print(f"🔹 Requête SQL Générée : {sql_query}")

if __name__ == "__main__":
    main()
