import pandas as pd
import ast

# Load CSV and safely convert strings to Python lists with ast
df = pd.read_csv("./data/merged_products.csv")
df["additives_tags"] = df["additives_tags"].dropna().apply(ast.literal_eval)

# Combine all additive tags into one flat set
all_additives = set()
for tags in df["additives_tags"]:
    all_additives.update(tags)

# Sorted list
all_additives_sorted = sorted(all_additives)

print(all_additives_sorted)
print(len(all_additives_sorted))
