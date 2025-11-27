from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import utils
from Constants import CSV_FILE_NAME

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Default number of clusters if we don't have a per-category best_k
DEFAULT_K = 8

# Where visualize_k.py writes its per-category k summary
BASE_DIR = Path(__file__).resolve().parent
K_SELECTION_PATH = BASE_DIR / "reports" / "figures" / "k_selection_summary.csv"

# Nutritional features used for clustering (same as in visualize_k.py)
NUTRITIONAL_FEATURES = [
    "energy_kcal_100g",
    "fat_100g",
    "saturated_fat_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
]

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = utils.read_csv(CSV_FILE_NAME)

# All the categories in the dataframe
categories = df["source_category"].dropna().unique().tolist()

# ---------------------------------------------------------------------
# Load best_k per category (from visualize_k.py)
# ---------------------------------------------------------------------

def load_best_k_mapping(path: Path) -> dict[str, int]:
    """
    Load the per-category best_k from k_selection_summary.csv.

    Returns a dict:
        {source_category: best_k}

    Any NaN or < 2 values are ignored and will fall back to DEFAULT_K.
    """
    if not path.exists():
        print(
            f"[TrainKMeans] WARNING: {path} not found. "
            f"Using DEFAULT_K={DEFAULT_K} for all categories."
        )
        return {}

    k_df = pd.read_csv(path)

    if not {"source_category", "best_k"}.issubset(k_df.columns):
        print(
            f"[TrainKMeans] WARNING: {path} does not contain "
            "'source_category' and 'best_k' columns. "
            f"Using DEFAULT_K={DEFAULT_K} for all categories."
        )
        return {}

    best_k_map: dict[str, int] = {}

    for _, row in k_df.iterrows():
        cat = str(row["source_category"])
        best_k = row["best_k"]

        # Skip NaN or invalid k
        if pd.isna(best_k):
            continue

        k_int = int(best_k)
        if k_int < 2:
            continue

        best_k_map[cat] = k_int

    print("[TrainKMeans] Loaded best_k per category from", path)
    for cat, k in best_k_map.items():
        print(f"    {cat}: k = {k}")
    print()

    return best_k_map


best_k_map = load_best_k_mapping(K_SELECTION_PATH)

# ---------------------------------------------------------------------
# Train one KMeans per category, using best_k when available
# ---------------------------------------------------------------------

for category in categories:
    df_category = df[df["source_category"] == category].copy()

    # Keep only rows with all nutritional features
    df_category = df_category.dropna(subset=NUTRITIONAL_FEATURES)
    if df_category.empty:
        print(
            f"[TrainKMeans] Category '{category}' has no rows with complete "
            "nutritional data. Skipping."
        )
        continue

    X = df_category[NUTRITIONAL_FEATURES].values

    # Decide k for this category
    n_clusters = best_k_map.get(category, DEFAULT_K)

    print(f"[TrainKMeans] Training KMeans for category '{category}' with k={n_clusters}")

    # Normalize X to mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Save models/scalers
    joblib.dump(kmeans, BASE_DIR / "models" / f"kmeans_{category}.pkl")
    joblib.dump(scaler, BASE_DIR / "models" / f"scaler_{category}.pkl")

    print(f"[TrainKMeans] Model for category '{category}' trained successfully!\n")

# ---------------------------------------------------------------------
# Assign clusters to the full CSV using the trained models
# ---------------------------------------------------------------------

print("[TrainKMeans] Assigning cluster labels to all rows...")

df["cluster"] = np.nan  # initialize

for category in categories:
    df_category = df[df["source_category"] == category].copy()
    df_category = df_category.dropna(subset=NUTRITIONAL_FEATURES)

    if df_category.empty:
        print(
            f"[TrainKMeans] Category '{category}' has no rows with complete "
            "nutritional data for prediction. Skipping."
        )
        continue

    scaler_path = BASE_DIR / "models" / f"scaler_{category}.pkl"
    kmeans_path = BASE_DIR / "models" / f"kmeans_{category}.pkl"

    if not scaler_path.exists() or not kmeans_path.exists():
        print(
            f"[TrainKMeans] WARNING: Missing model or scaler for category "
            f"'{category}'. Skipping cluster assignment."
        )
        continue

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)

    X = df_category[NUTRITIONAL_FEATURES].values
    X_scaled = scaler.transform(X)
    predicted_clusters = kmeans.predict(X_scaled)

    # Write back into the main df
    df.loc[df_category.index, "cluster"] = predicted_clusters
    print(f"[TrainKMeans] Elements from category '{category}' clustered successfully.")

# Save updated CSV
out_path = BASE_DIR / "data" / "merged_products_clustered.csv"
df.to_csv(out_path, index=False)
print(f"[TrainKMeans] Clustered CSV saved to {out_path}")