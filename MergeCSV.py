import pandas as pd
import os
from glob import glob

# Set the directory where your CSVs are stored
csv_dir = "./data/CSV_files"

# Get all CSV file paths in that folder
csv_files = glob(os.path.join(csv_dir, "*.csv"))

# Load and store each CSV file as a DataFrame
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df["source_category"] = os.path.basename(file).replace(".csv", "")  # optional: track original category
    dfs.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(dfs, ignore_index=True)

# Move "source_category" to the first column
cols = merged_df.columns.tolist()
if "source_category" in cols:
    cols = ["source_category"] + [col for col in cols if col != "source_category"]
    merged_df = merged_df[cols]

# Filtering duplicates
merged_df_clean = merged_df.drop_duplicates(subset="code")

# Reset index
merged_df_clean.reset_index(drop=True, inplace=True)

# Save the final merged CSV
merged_path = "./data/merged_products.csv"
merged_df_clean.to_csv(merged_path, index=False)

print(f"Merged {len(csv_files)} files into {merged_path}")
print(f"Total products: {len(merged_df)}")
