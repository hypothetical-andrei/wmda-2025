# [Querying a Sqlite database to retrieve user transaction data.]
import sqlite3

# Step 1: Connect to SQLite database (creates a new file if it doesn't exist)
conn = sqlite3.connect("transactions.db")
cursor = conn.cursor()

# Step 2: Create a transactions table (if it doesn't exist)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        amount REAL,
        transaction_type TEXT,
        date TEXT
    )
''')

# Step 3: Insert sample data (only if the table is empty)
cursor.execute("SELECT COUNT(*) FROM transactions")
if cursor.fetchone()[0] == 0:
    sample_data = [
        (1, 250.75, 'deposit', '2024-02-23'),
        (1, -100.50, 'withdrawal', '2024-02-24'),
        (2, 500.00, 'deposit', '2024-02-20'),
        (3, -75.00, 'withdrawal', '2024-02-21'),
    ]
    cursor.executemany("INSERT INTO transactions (user_id, amount, transaction_type, date) VALUES (?, ?, ?, ?)", sample_data)
    conn.commit()

# Step 4: Query and retrieve user transactions (e.g., transactions for user_id = 1)
user_id = 1
cursor.execute("SELECT * FROM transactions WHERE user_id = ?", (user_id,))
transactions = cursor.fetchall()

# Step 5: Display results
print(f"Transactions for User {user_id}:\n")
for txn in transactions:
    print(f"ID: {txn[0]}, Amount: {txn[2]}, Type: {txn[3]}, Date: {txn[4]}")

# Close connection
conn.close()
