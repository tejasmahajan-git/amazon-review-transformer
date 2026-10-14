
import gzip
import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

REVIEWS_FILE = Path("data/amazon_electronics_reviews_sample.csv")
OUTPUT_FILE = Path("data/amazon_electronics_product_metadata.csv")

URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/"
    "amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz"
)

reviews = pd.read_csv(REVIEWS_FILE, usecols=["product_id"])
product_ids = set(
    reviews["product_id"].dropna().astype(str)
)

matched = {}

print(f"Looking for {len(product_ids):,} products.")
print("Streaming the official compressed metadata file...")

with requests.get(
    URL,
    stream=True,
    timeout=(30, 180),
) as response:
    response.raise_for_status()
    response.raw.decode_content = False

    with gzip.GzipFile(fileobj=response.raw) as source:
        # Approximate total: the full category has around
        # 1.6 million product metadata records.
        for line in tqdm(source, total=1_600_000, unit="records"):
            if not line:
                continue

            item = json.loads(line)
            product_id = item.get("parent_asin")

            if product_id in product_ids:
                matched[product_id] = {
                    "product_id": product_id,
                    "product_title": item.get("title"),
                    "brand": item.get("store"),
                    "main_category": item.get("main_category"),
                    "average_rating": item.get("average_rating"),
                    "rating_number": item.get("rating_number"),
                    "price": item.get("price"),
                    "features": json.dumps(item.get("features")),
                    "description": json.dumps(item.get("description")),
                    "details": json.dumps(item.get("details")),
                }

                if len(matched) == len(product_ids):
                    print("\nAll requested products found.")
                    break

df = pd.DataFrame(matched.values())
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

total = len(product_ids)
count = len(matched)

print("\nFinished.")
print(f"Products matched: {count:,}/{total:,}")
print(f"Match rate: {count / total:.1%}" if total else "No IDs found.")
print(f"Saved to: {OUTPUT_FILE.resolve()}")
