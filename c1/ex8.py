# [Encoding gender (male/female) as binary values.]
import pandas as pd

# Step 1: Create a sample dataset with gender column
data = {
    "PersonID": [1, 2, 3, 4, 5],
    "Name": ["John", "Emma", "Liam", "Sophia", "Noah"],
    "Gender": ["Male", "Female", "Male", "Female", "Male"]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Step 2: Encode gender as binary values (Male=0, Female=1)
df["Gender_Binary"] = df["Gender"].map({"Male": 0, "Female": 1})

print("\nDataset After Encoding Gender as Binary:\n")
print(df)
