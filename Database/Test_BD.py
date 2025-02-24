import sqlite3

# Connexion à la base de données SQLite
conn = sqlite3.connect('Database/BD.db')
cursor = conn.cursor()

# Requête SQL à tester
query = "SELECT * FROM employees WHERE department_id = 10 AND salary > 5000;"

# Exécution de la requête
cursor.execute(query)

# Récupération des résultats
results = cursor.fetchall()

# Affichage des résultats
if results:
    for row in results:
        print(row)  # Chaque ligne correspond à un employé qui répond aux critères
else:
    print("Aucun résultat trouvé pour cette requête.")

# Fermeture de la connexion à la base de données
conn.close()