import json
import pandas as pd

def json_to_csv(json_path, csv_path):
    # Open the json file, read it and load it in 'data' dictionary
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    # Save in the dictionary all the values from data
    products = data.get("products", [])
    cleaned = []

    # Get the desired fields for each product
    for product in products:
        nutr = product.get("nutriments", {})
        cleaned.append({
            "product_name": product.get("product_name", ""),
            "code": product.get("code", ""),
            "categories_tags": product.get("categories_tags", []),
            "additives_n": product.get("additives_n", 0),
            "additives": product.get("additives", []),
            "additives_tags": product.get("additives_tags", []),
            "energy_kcal_100g": nutr.get("energy-kcal_100g"),
            "fat_100g": nutr.get("fat_100g"),
            "saturated_fat_100g": nutr.get("saturated-fat_100g"),
            "sugars_100g": nutr.get("sugars_100g"),
            "fiber_100g": nutr.get("fiber_100g"),
            "proteins_100g": nutr.get("proteins_100g"),
            "salt_100g": nutr.get("salt_100g")
        })

    # Transform into Pandas Dataframe
    df = pd.DataFrame(cleaned)

    # Filter out products missing core values
    df = df.dropna(subset=[
        "energy_kcal_100g", "fat_100g", "saturated_fat_100g",
        "sugars_100g", "proteins_100g", "salt_100g"
    ])

    # Transform into CSV

    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} cleaned products to {csv_path}")

# Calling the function with all the categories
input_prefix = "./data/JSON_files/"
output_prefix = "./data/CSV_files/"

categories = [
    "breads",
    "breakfast-cereals",
    "cheeses",
    "ice-creams",
    "meals-with-meat",
    "microwave-meals",
    "plant-based-foods",
    "snacks",
    "soft-drinks",
    "yogurts"
]

for category in categories:
    json_to_csv(f"{input_prefix}{category}.json", f"{output_prefix}{category}.csv")