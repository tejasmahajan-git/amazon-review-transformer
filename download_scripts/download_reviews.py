
import json
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://huggingface.co/datasets/"
    "McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/Electronics.jsonl"
)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "amazon_electronics_reviews_sample.csv"
MAX_REVIEWS = 5_000

reviews = []

print("Connecting to Amazon Reviews 2023...")

with requests.get(URL, stream=True, timeout=(20, 120)) as response:
    response.raise_for_status()

    # Prevent requests from automatically decompressing the stream.
    response.raw.decode_content = False

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        review = json.loads(line)

        reviews.append({
            "product_id": review.get("parent_asin"),
            "rating": review.get("rating"),
            "review_title": review.get("title"),
            "review_text": review.get("text"),
            "timestamp": review.get("timestamp"),
            "helpful_votes": review.get("helpful_vote"),
            "verified_purchase": review.get("verified_purchase"),
        })

        if len(reviews) >= MAX_REVIEWS:
            break

df = pd.DataFrame(reviews)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Saved {len(df):,} reviews to {OUTPUT_FILE}")
print(df[["rating", "review_title", "review_text"]].head().to_string(index=False))
