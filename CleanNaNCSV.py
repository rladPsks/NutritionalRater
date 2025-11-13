"""import pandas as pd

df = pd.read_csv("./data/merged_products_randomized.csv")

# Check if there are any NaN at all
print(df.isna().any().any())  # True → there are NaN; False → no NaN

# See which columns have NaN
print(df.isna().sum())"""

"""import pandas as pd

df = pd.read_csv("./data/merged_products_randomized.csv")

# Filter rows where fiber_100g is NaN
nan_fiber_df = df[df["fiber_100g"].isna()]

# Save them into a new CSV
nan_fiber_df.to_csv("./data/products_missing_fiber.csv", index=False)

print(f"Saved {len(nan_fiber_df)} products with NaN in 'fiber_100g' to products_missing_fiber.csv")"""

import pandas as pd

df = pd.read_csv("./data/merged_products_randomized.csv")

# Drop rows with NaN in 'fiber_100g'
df_cleaned = df.dropna(subset=["fiber_100g"])

# Save the cleaned version
df_cleaned.to_csv("./data/merged_products_no_nan_fiber.csv", index=False)

print(f"Removed {len(df) - len(df_cleaned)} rows with NaN in 'fiber_100g'.")
print(f"New file saved as merged_products_no_nan_fiber.csv")

