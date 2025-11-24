# visualize_k.py
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES = [
    "energy_kcal_100g", "fat_100g", "saturated_fat_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g"
]

def safe_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def evaluate_k(X_scaled, k_min, k_max):
    results = []
    for k in range(k_min, k_max + 1):
        if X_scaled.shape[0] <= k:
            # Not enough samples to form k clusters
            results.append({"k": k, "inertia": np.nan, "silhouette": np.nan})
            continue
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        # silhouette requires at least 2 clusters and > k samples
        sil = np.nan
        if len(set(labels)) > 1 and X_scaled.shape[0] > k:
            sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": inertia, "silhouette": sil})
    return pd.DataFrame(results)

def pick_best_k(scores_df):
    # Strategy: choose k with the highest silhouette; fallback to elbow (inertia knee) if all NaN
    if scores_df["silhouette"].notna().any():
        s = scores_df.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False)
        return int(s.iloc[0]["k"]), {"criterion": "silhouette", "value": float(s.iloc[0]["silhouette"])}
    # crude elbow fallback: pick the k with the biggest relative drop in inertia
    d = scores_df.dropna(subset=["inertia"]).copy()
    d["delta"] = d["inertia"].shift(1) - d["inertia"]
    d = d.dropna(subset=["delta"]).sort_values("delta", ascending=False)
    if not d.empty:
        return int(d.iloc[0]["k"]), {"criterion": "elbow_delta", "value": float(d.iloc[0]["delta"])}
    return None, {"criterion": "none", "value": None}

def plot_lines(cat, scores_df, out_dir):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/merged_products.csv",
                        help="Path to the merged products CSV (pre- or post-clustered is fine)")
    parser.add_argument("--out", default="reports/figures",
                        help="Output directory for figures and summary")
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = safe_numeric(df, FEATURES)

    if "source_category" not in df.columns:
        raise ValueError("CSV must contain a 'source_category' column.")

    rows = []
    for cat, g in df.groupby("source_category"):
        g2 = g.dropna(subset=FEATURES).copy()
        if len(g2) < max(3, args.kmin + 1):
            # too few rows to evaluate clustering meaningfully
            rows.append({"source_category": cat, "best_k": np.nan, "criterion": "insufficient_data", "value": np.nan})
            continue

        X = g2[FEATURES].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        scores = evaluate_k(Xs, args.kmin, args.kmax)
        plot_lines(cat, scores, args.out)

        best_k, info = pick_best_k(scores)
        rows.append({"source_category": cat, "best_k": best_k, "criterion": info["criterion"], "value": info["value"]})

    summary = pd.DataFrame(rows).sort_values("source_category")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "k_selection_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(f"Figures saved to {out_dir}")

if __name__ == "__main__":
    main()