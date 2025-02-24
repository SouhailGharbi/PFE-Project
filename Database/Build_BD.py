import sqlite3

# Connexion à la base de données SQLite (ou création si elle n'existe pas)
conn = sqlite3.connect('Database/BD.db')
cursor = conn.cursor()

# 1. Création des tables

# Table Employees
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    department_id INTEGER,
    salary REAL,
    hire_date TEXT
);
''')

# Table Departments
cursor.execute('''
CREATE TABLE IF NOT EXISTS departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT
);
''')

# Table Customers
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT,
    city TEXT,
    country TEXT
);
''')

# Table Orders
cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date TEXT,
    total_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);
''')

# Table Products
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    price REAL,
    stock_quantity INTEGER
);
''')

# Table Order_Items
cursor.execute('''
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    price REAL,
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
''')

# Table Suppliers
cursor.execute('''
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id INTEGER PRIMARY KEY,
    supplier_name TEXT,
    contact_name TEXT,
    contact_email TEXT
);
''')

# Table Product_Suppliers
cursor.execute('''
CREATE TABLE IF NOT EXISTS product_suppliers (
    product_id INTEGER,
    supplier_id INTEGER,
    PRIMARY KEY (product_id, supplier_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers (supplier_id)
);
''')

# Table Sales
cursor.execute('''
CREATE TABLE IF NOT EXISTS sales (
    sale_id INTEGER PRIMARY KEY,
    employee_id INTEGER,
    product_id INTEGER,
    sale_date TEXT,
    quantity_sold INTEGER,
    FOREIGN KEY (employee_id) REFERENCES employees (employee_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
''')

# Table Employees_Training
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees_training (
    employee_id INTEGER,
    training_name TEXT,
    completion_date TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
);
''')

# 2. Insertion des données

# Insertion dans la table Departments
cursor.executemany('''
INSERT INTO departments (department_id, department_name)
VALUES (?, ?);
''', [
    (10, 'Sales'),
    (20, 'HR'),
    (30, 'IT')
])

# Insertion dans la table Employees
cursor.executemany('''
INSERT INTO employees (employee_id, first_name, last_name, department_id, salary, hire_date)
VALUES (?, ?, ?, ?, ?, ?);
''', [
    (1, 'John', 'Doe', 10, 6000, '2020-01-15'),
    (2, 'Jane', 'Smith', 20, 4500, '2019-03-22')
])

# Insertion dans la table Customers
cursor.executemany('''
INSERT INTO customers (customer_id, customer_name, city, country)
VALUES (?, ?, ?, ?);
''', [
    (1, 'ABC Corp', 'New York', 'USA'),
    (2, 'XYZ Ltd', 'London', 'UK')
])

# Insertion dans la table Orders
cursor.executemany('''
INSERT INTO orders (order_id, customer_id, order_date, total_amount)
VALUES (?, ?, ?, ?);
''', [
    (1, 1, '2025-02-20', 1500),
    (2, 2, '2025-02-21', 2000)
])

# Insertion dans la table Products
cursor.executemany('''
INSERT INTO products (product_id, product_name, price, stock_quantity)
VALUES (?, ?, ?, ?);
''', [
    (1, 'Laptop', 1000, 50),
    (2, 'Mouse', 20, 200)
])

# Insertion dans la table Order_Items
cursor.executemany('''
INSERT INTO order_items (order_item_id, order_id, product_id, quantity, price)
VALUES (?, ?, ?, ?, ?);
''', [
    (1, 1, 1, 1, 1000),
    (2, 2, 2, 2, 20)
])

# Insertion dans la table Suppliers
cursor.executemany('''
INSERT INTO suppliers (supplier_id, supplier_name, contact_name, contact_email)
VALUES (?, ?, ?, ?);
''', [
    (1, 'TechCorp', 'Alice Brown', 'alice@techcorp.com'),
    (2, 'GadgetsWorld', 'Bob White', 'bob@gadgetsworld.com')
])

# Insertion dans la table Product_Suppliers
cursor.executemany('''
INSERT INTO product_suppliers (product_id, supplier_id)
VALUES (?, ?);
''', [
    (1, 1),
    (2, 2)
])

# Insertion dans la table Sales
cursor.executemany('''
INSERT INTO sales (sale_id, employee_id, product_id, sale_date, quantity_sold)
VALUES (?, ?, ?, ?, ?);
''', [
    (1, 1, 1, '2025-02-20', 10),
    (2, 2, 2, '2025-02-21', 15)
])

# Insertion dans la table Employees_Training
cursor.executemany('''
INSERT INTO employees_training (employee_id, training_name, completion_date)
VALUES (?, ?, ?);
''', [
    (1, 'Data Science', '2025-01-10'),
    (2, 'Machine Learning', '2025-02-15')
])

# 3. Sauvegarde des modifications et fermeture de la connexion
conn.commit()

# Fermeture de la connexion à la base de données
conn.close()

print("Base de données créée et chargée avec succès.")
