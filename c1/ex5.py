# [Cleaning a customer database by removing duplicate records.]
import pandas as pd

# Step 1: Create a sample customer database with duplicates
data = {
    "customer_id": [101, 102, 103, 101, 104, 102],
    "name": ["Alice", "Bob", "Charlie", "Alice", "David", "Bob"],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com",
              "alice@example.com", "david@example.com", "bob@example.com"]
}

df = pd.DataFrame(data)

print("Original Customer Database:\n")
print(df)

# Step 2: Remove duplicate records based on all columns
df_cleaned = df.drop_duplicates()

print("\nDatabase After Removing Exact Duplicates:\n")
print(df_cleaned)

# Step 3: Remove duplicates based on specific columns (e.g., customer_id)
df_cleaned = df.drop_duplicates(subset=["customer_id"], keep="first")

print("\nDatabase After Removing Duplicates Based on Customer ID:\n")
print(df_cleaned)
