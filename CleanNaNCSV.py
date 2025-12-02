import pandas as pd

df = pd.read_csv("./data/merged_products_randomized.csv")

# Drop rows with NaN in 'fiber_100g'
df_cleaned = df.dropna(subset=["fiber_100g"])

# Save the cleaned version
df_cleaned.to_csv("./data/merged_products_no_nan_fiber.csv", index=False)

print(f"Removed {len(df) - len(df_cleaned)} rows with NaN in 'fiber_100g'.")
print(f"New file saved as merged_products_no_nan_fiber.csv")

