import pandas as pd

def recommend_alternatives(df: pd.DataFrame, product_name: str, top_k: int = 5) -> pd.DataFrame:
    """
    Recommend healthier alternatives within the same category.

    Parameters
    ----------
    df : DataFrame
        DataFrame with at least 'product_name', 'source_category', 'rating'.
    product_name : str
        Name of the product to find alternatives for.
    top_k : int
        Number of alternatives to return.

    Returns
    -------
    DataFrame
        Top-k healthier alternatives with product_name and rating.
    """
    # Drop rows with missing product_name so we don't recommend NaN names
    df_clean = df.dropna(subset=["product_name"]).copy()

    # 1) Find the target product row
    target_rows = df_clean[df_clean["product_name"] == product_name]
    if len(target_rows) == 0:
        raise ValueError(f"Product '{product_name}' not found in DataFrame.")

    target = target_rows.iloc[0]
    target_cat = target.get("source_category")
    target_score = target.get("rating")

    # 2) Filter products in the same category
    same_cat = df_clean[df_clean["source_category"] == target_cat].copy()

    # (Optional but sensible) remove the target product itself from candidates
    same_cat = same_cat[same_cat.index != target.name]

    # 3) Keep only products with a higher rating
    better = same_cat[same_cat["rating"] > target_score]

    # 4) Sort by rating descending and return top_k
    better = better.sort_values("rating", ascending=False)

    return better.head(top_k)[["product_name", "rating"]]
