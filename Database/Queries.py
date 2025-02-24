import sqlite3

# Connexion à la base de données
conn = sqlite3.connect("Database/BD.db")
cursor = conn.cursor()

# 📌 Exécuter une jointure : Récupérer les employés avec leur département
cursor.execute("""
    SELECT employees.first_name, employees.last_name, departments.department_name
    FROM employees
    INNER JOIN departments ON employees.department_id = departments.department_id
""")

# Afficher les résultats
results = cursor.fetchall()
for row in results:
    print(row)

# Fermer la connexion
conn.close()
