import pandas as pd
import utils
from Constants import CSV_FILE_NAME

# This function is now in utils
"""
# Read the CSV indexing by barcode strings
df = pd.read_csv(CSV_FILE_NAME, dtype={"code": "string"})
# Index by code without droping the column in the CSV
df = df.set_index("code", drop=False)
# Make sure that there are no duplicate codes
assert df.index.is_unique, "Duplicate barcodes found in the dataset"
"""
df = utils.read_csv(CSV_FILE_NAME)

if __name__ == "__main__":
    # Ask the user to input the barcode
    print("Input the barcode of the product you want to scan:")


    # Read the barcode
    barcode = input().strip()

    # If the barcode is in in the database
    if barcode in df.index:

        # Locate the product
        product = df.loc[barcode]

        # Print the product's info
        print("\nProduct info:")
        print(f"    Name: {product['product_name']}")
        print(f"    Category: {product['source_category']}")
        print(f"    Number of additives: {product['additives_n']}")

        if product['additives_n'] > 0:
            print(f"    Additives tags: {product['additives_tags']}")
        

        # TODO: Here we could call the formula function, which function should output depending on the category/categories
        # We are analyzing if the amount of x nutrient is rated as bad, could be better or good

        print(f"    Final rating: {product['rating']}\n")

        # TODO: Apply the already trained K-Means
        

    else:
        print("Sorry! Your product is not in the database")