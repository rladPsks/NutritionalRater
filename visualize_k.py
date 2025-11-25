# visualize_k.py
"""
Functions
1) Evaluate different KMeans cluster counts (k) per product category
   using inertia (elbow) and silhouette scores.
2) Visualize the currently trained KMeans clusters and their centers
   in 2D using PCA.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Nutritional features used for clustering and visualization
FEATURES = [
    "energy_kcal_100g", "fat_100g", "saturated_fat_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g"
]


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Ensure a set of columns are numeric.

    Any non-numeric value is coerced to NaN instead of raising an error.
    Returns a copy of the original DataFrame.
    """
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def evaluate_k(X_scaled: np.ndarray, k_min: int, k_max: int) -> pd.DataFrame:
    """
    For a given scaled feature matrix X_scaled, evaluate KMeans performance
    for k in [k_min, k_max].

    Returns a DataFrame with columns:
        - k: number of clusters
        - inertia: within-cluster sum of squares (lower is better, but looking for slowed down decrease)
        - silhouette: silhouette score (higher is better)
    """
    results = []
    for k in range(k_min, k_max + 1):
        # Not enough samples to form k clusters
        if X_scaled.shape[0] <= k:
            results.append({"k": k, "inertia": np.nan, "silhouette": np.nan})
            continue

        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_

        # Silhouette requires at least 2 clusters and more samples than k
        sil = np.nan
        if len(set(labels)) > 1 and X_scaled.shape[0] > k:
            sil = silhouette_score(X_scaled, labels)

        results.append({"k": k, "inertia": inertia, "silhouette": sil})

    return pd.DataFrame(results)


def pick_best_k(scores_df: pd.DataFrame) -> tuple[int | None, dict]:
    """
    Given a scores DataFrame from evaluate_k(), choose the "best" k.

    Strategy:
        1) If any silhouette scores are available, pick the k with the
           highest silhouette score.
        2) Otherwise, fall back to a crude elbow heuristic:
           pick the k with the largest drop in inertia compared to k-1.
        3) If nothing is available, return (None, info).

    Returns:
        best_k: int or None
        info: dict with keys 'criterion' and 'value'
    """
    # Prefer silhouette if available
    if scores_df["silhouette"].notna().any():
        s = scores_df.dropna(subset=["silhouette"]).sort_values(
            "silhouette", ascending=False
        )
        return int(s.iloc[0]["k"]), {
            "criterion": "silhouette",
            "value": float(s.iloc[0]["silhouette"]),
        }

    # Fallback: largest drop in inertia (elbow-style)
    d = scores_df.dropna(subset=["inertia"]).copy()
    d["delta"] = d["inertia"].shift(1) - d["inertia"]
    d = d.dropna(subset=["delta"]).sort_values("delta", ascending=False)
    if not d.empty:
        return int(d.iloc[0]["k"]), {
            "criterion": "elbow_delta",
            "value": float(d.iloc[0]["delta"]),
        }

    # No usable information
    return None, {"criterion": "none", "value": None}


def plot_lines(cat: str, scores_df: pd.DataFrame, out_dir: str | Path) -> None:
    """
    For a single category, plot:
        - k vs inertia (elbow plot)
        - k vs silhouette score

    Images are saved in out_dir as:
        <category>_elbow.png
        <category>_silhouette.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inertia (Elbow)
    plt.figure()
    plt.plot(scores_df["k"], scores_df["inertia"], marker="o")
    plt.title(f"Elbow (Inertia) — {cat}")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(out_dir / f"{cat}_elbow.png", dpi=160)
    plt.close()

    # Silhouette
    plt.figure()
    plt.plot(scores_df["k"], scores_df["silhouette"], marker="o")
    plt.title(f"Silhouette Score — {cat}")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig(out_dir / f"{cat}_silhouette.png", dpi=160)
    plt.close()


def plot_trained_clusters(
    df: pd.DataFrame,
    models_dir: str | Path = "models",
    out_dir: str | Path = "reports/cluster_plots",
) -> None:
    """
    Visualize the currently trained (k=8) KMeans clusters per category.

    For each source_category:
        - Load the saved scaler and kmeans model
          (scaler_{category}.pkl, kmeans_{category}.pkl).
        - Scale the category's data with the saved scaler.
        - Project data + cluster centers to 2D with PCA.
        - Plot points colored by cluster and show centers as "X".
        - Save to: reports/cluster_plots/<category>_clusters.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "source_category" not in df.columns:
        raise ValueError("CSV must contain a 'source_category' column.")

    for cat, g in df.groupby("source_category"):
        scaler_path = Path(models_dir) / f"scaler_{cat}.pkl"
        kmeans_path = Path(models_dir) / f"kmeans_{cat}.pkl"

        if not scaler_path.exists() or not kmeans_path.exists():
            print(f"[plot_trained_clusters] No model found for '{cat}', skipping.")
            continue

        # Keep only rows with all nutritional features present
        g2 = g.dropna(subset=FEATURES).copy()
        if len(g2) < 3:
            print(f"[plot_trained_clusters] Not enough data for '{cat}', skipping.")
            continue

        # Load trained scaler & model
        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)

        # Scale using the trained scaler
        X = g2[FEATURES].values
        X_scaled = scaler.transform(X)

        # Predict clusters with the current trained model
        labels = kmeans.predict(X_scaled)

        # PCA to 2D for visualization (fit on scaled data)
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)

        # Transform cluster centers to the scaled feature space
        centers_scaled = kmeans.cluster_centers_
        centers_2d = pca.transform(centers_scaled)

        # Plot data points and cluster centers
        plt.figure()
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, alpha=0.6)
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            marker="X",
            s=160,
            edgecolor="black",
        )
        plt.title(f"Clusters for '{cat}' (k={kmeans.n_clusters})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()

        out_path = out_dir / f"{cat}_clusters.png"
        plt.savefig(out_path, dpi=160)
        plt.close()

        print(f"[plot_trained_clusters] Saved cluster plot for '{cat}' → {out_path}")


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/merged_products.csv",
        help="Path to the merged products CSV (pre- or post-clustered is fine).",
    )
    parser.add_argument(
        "--out",
        default="reports/figures",
        help="Output directory for k-selection figures and summary CSV.",
    )
    parser.add_argument("--kmin", type=int, default=2, help="Minimum k to test.")
    parser.add_argument("--kmax", type=int, default=10, help="Maximum k to test.")
    args = parser.parse_args()

    # Load raw data
    df = pd.read_csv(args.csv)
    df = safe_numeric(df, FEATURES)

    if "source_category" not in df.columns:
        raise ValueError("CSV must contain a 'source_category' column.")

    rows = []

    # Evaluate k for each product category separately
    for cat, g in df.groupby("source_category"):
        g2 = g.dropna(subset=FEATURES).copy()
        if len(g2) < max(3, args.kmin + 1):
            # Too few rows to evaluate clustering meaningfully
            rows.append(
                {
                    "source_category": cat,
                    "best_k": np.nan,
                    "criterion": "insufficient_data",
                    "value": np.nan,
                }
            )
            continue

        X = g2[FEATURES].values

        # Scale features before KMeans
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Compute inertia and silhouette for multiple k
        scores = evaluate_k(Xs, args.kmin, args.kmax)

        # Save elbow & silhouette curves
        plot_lines(cat, scores, args.out)

        # Pick best k using silhouette / elbow
        best_k, info = pick_best_k(scores)
        rows.append(
            {
                "source_category": cat,
                "best_k": best_k,
                "criterion": info["criterion"],
                "value": info["value"],
            }
        )

    # Save per-category best k summary
    summary = pd.DataFrame(rows).sort_values("source_category")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "k_selection_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(f"Figures saved to {out_dir}")

    # Also show what the currently trained models look like in 2D
    print("\nNow plotting trained clusters with current k values...")
    plot_trained_clusters(df, models_dir="models", out_dir="reports/cluster_plots")


if __name__ == "__main__":
    main()