from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import utils
import joblib
from Constants import CSV_FILE_NAME

N_CLUSTERS = 8

df = utils.read_csv(CSV_FILE_NAME)

# All the categories in the dataframe
categories = df['source_category'].dropna().unique().tolist()

# Nutritional features in the dataframe
nutritional_features = [
    "energy_kcal_100g", "fat_100g", "saturated_fat_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g"
]

for category in categories:
    df_category = df[df['source_category'] == category]

    X = df_category[nutritional_features]

    # Normalize X to transform every feature into a mean = 0, std = 1 distribution
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the k-means
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    kmeans.fit(X_scaled)

    # Save the models and Scalers to memory after being trained to use them later
    joblib.dump(kmeans, f"./models/kmeans_{category}.pkl")
    joblib.dump(scaler, f"./models/scaler_{category}.pkl")

    print(f"Model for category {category} trained successfully!")


# Add the cluster assigned to the CSV file:

# Load your full CSV again. Necessary?
#df = utils.read_csv(CSV_FILE_NAME)

cluster_column = []

for category in categories:
    df_category = df[df["source_category"] == category]
    X = df_category[nutritional_features]

    # Load trained models
    scaler = joblib.load(f"./models/scaler_{category}.pkl")
    kmeans = joblib.load(f"./models/kmeans_{category}.pkl")

    # Normalize the values in all the dataframe
    X_scaled = scaler.transform(X)
    # Predict the clusters of all the values in that category (the dataframe)
    predicted_clusters = kmeans.predict(X_scaled)

    # Assign cluster IDs to the matching rows
    df.loc[df["source_category"] == category, "cluster"] = predicted_clusters

    print(f"Elements from the category {category} clustered successfully!")

# Save updated CSV
df.to_csv("./data/merged_products_clustered.csv", index=False)
print("Clustered CSV saved.")