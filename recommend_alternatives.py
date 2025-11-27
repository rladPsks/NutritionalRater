import pandas as pd

def recommend_alternatives(df: pd.DataFrame, product_name: str, top_k: int = 5) -> pd.DataFrame:
    """
    Recommend healthier alternatives within the same category.

    Parameters
    ----------
    df : DataFrame
        DataFrame with at least 'product_name', 'source_category', 'health_score'.
    product_name : str
        Name of the product to find alternatives for.
    top_k : int
        Number of alternatives to return.

    Returns
    -------
    DataFrame
        Top-k healthier alternatives with product_name and health_score.
    """
    # 1) Find the target product row
    target_rows = df[df["product_name"] == product_name]
    if len(target_rows) == 0:
        raise ValueError(f"Product '{product_name}' not found in DataFrame.")

    target = target_rows.iloc[0]
    target_cat = target.get("source_category")
    target_score = target.get("health_score")

    # 2) Filter products in the same category
    same_cat = df[df["source_category"] == target_cat].copy()

    # 3) Keep only products with a higher health score
    better = same_cat[same_cat["health_score"] > target_score]

    # 4) Sort by health_score descending and return top_k
    better = better.sort_values("health_score", ascending=False)

    return better.head(top_k)[["product_name", "health_score"]]
