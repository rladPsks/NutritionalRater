import pandas as pd
from pathlib import Path

from Constants import CSV_FILE_NAME
import utils

from TrainKMeans import train_models_with_best_k   # Must be implemented in TrainKMeans
from score_products import score_dataframe


def ensure_models_are_up_to_date():
    """
    Checks and loads the best-k summary, then retrains the KMeans models.
    This is executed once at startup before the user interacts with the system.
    """
    summary_path = Path("reports/figures/k_selection_summary.csv")

    if not summary_path.exists():
        print("[INFO] No k-selection summary found. Please run visualize_k.py first.\n")
        return

    print("[INFO] Updating clustering models based on best-k values...")
    train_models_with_best_k()
    print("[INFO] Models retrained successfully.\n")


def load_final_dataframe():
    """
    Loads the merged + rated + clustered CSV.
    If 'rating' is missing, compute all scores.
    """
    df = utils.read_csv(CSV_FILE_NAME)

    if "rating" not in df.columns:
        print("[INFO] No rating found – computing nutritional scores for all products...")
        df = score_dataframe(df)
        df.to_csv(CSV_FILE_NAME, index=False)
        print("[INFO] Ratings saved.\n")

    return df


def main():

    # --- Step 0: Ensure models are up to date before showing menu ---
    ensure_models_are_up_to_date()

    # --- Step 1: Banner ---
    print("===================================")
    print("   Nutritional Rater — Main Menu   ")
    print("===================================\n")

    # --- Step 2: Load data ---
    df = load_final_dataframe()
    df = df.set_index("code", drop=False)

    # --- Step 3: Get user input ---
    print("Input the barcode of the product you want to scan:")
    barcode = input().strip()

    if barcode not in df.index:
        print("\nSorry! Your product is not in the database.\n")
        return

    product = df.loc[barcode]

    # --- Step 4: Show product info ---
    print("\n=== Product Information ===")
    print(f"Name:              {product['product_name']}")
    print(f"Category:          {product['source_category']}")
    print(f"Rating (0–100):    {product['rating']}")
    print(f"Additives count:   {product['additives_n']}")
    if product['additives_n'] > 0:
        print(f"Additives tags:    {product['additives_tags']}")

    # --- Step 5: TODO placeholder for alternatives ---
    print("\n=== Healthier Alternatives ===")
    # TODO: integrate recommend_alternatives() once implemented.\n")
    

    print("Done.\n")


if __name__ == "__main__":
    main()