from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 📌 Charger le modèle fine-tuné et le tokenizer
model_name = "./fine_tuned_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 📌 Ajouter le token de padding
tokenizer.pad_token = tokenizer.eos_token

# 📌 Fonction pour générer une requête SQL
def generate_sql(query):
    prompt = f"Question: {query}"  # Format structuré \nRequête SQL:
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():  
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  
            max_length=100,  
            num_return_sequences=1,  
            pad_token_id=tokenizer.eos_token_id
        )

    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    #  Supprimer tout sauf la requête SQL
    #if "Requête SQL:" in generated_sql:
        #generated_sql = generated_sql.split("Requête SQL:")[-1].strip()

    #  Vérifier si le modèle a généré du SQL valide
    if not generated_sql.strip().startswith("SELECT"):
        return " Le modèle n'a pas généré une requête SQL correcte."

    return generated_sql

#  Mode interactif
print("\n✅ Tape une question en langage naturel pour générer une requête SQL (ou 'exit' pour quitter) :")

while True:
    user_input = input("\n🔹 Question : ")
    if user_input.lower() == "exit":
        print("👋 Fin du test.")
        break
    
    generated_sql = generate_sql(user_input)
    print(f"🟢 Requête SQL générée : {generated_sql}")
