import os
import requests
import json
import time

# Categories you want to download
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

max_pages = 50
delay_seconds = 1
output_dir = "./data/JSON_files"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Use a realistic User-Agent to avoid being blocked by the API
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    )
}

for category in categories:
    print(f"\nDownloading category: {category}")
    all_products = []

    for page in range(1, max_pages + 1):
        url = f"https://world.openfoodfacts.org/category/{category}/{page}.json"
        print(f"  Fetching page {page}...")

        try:
            res = requests.get(url, headers=headers, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Request error on page {page}: {e}")
            break

        if res.status_code != 200 or not res.text.strip():
            print(f"Bad response on page {page}. Status: {res.status_code}")
            break

        try:
            data = res.json()
        except ValueError:
            print(f"Failed to parse JSON on page {page}. HTML/invalid JSON received.")
            break

        products = data.get("products", [])
        if not products:
            print(f"No more products found on page {page}.")
            break

        all_products.extend(products)
        print(f"Retrieved {len(products)} products (total: {len(all_products)})")
        time.sleep(delay_seconds)

    # Save results to JSON
    output_path = os.path.join(output_dir, f"{category}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"products": all_products}, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_products)} products to {output_path}")
