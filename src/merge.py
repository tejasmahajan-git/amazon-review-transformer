import pandas as pd

reviews = pd.read_csv(
    "C:/Users/tejas/Projects/amazon-customer-intelligence/data/amazon_electronics_reviews_sample.csv"
)


products = pd.read_csv(
    "C:/Users/tejas/Projects/amazon-customer-intelligence/data/amazon_electronics_product_metadata.csv"
)

combined = reviews.merge(
    products,
    on="product_id",
    how="left",
    validate="many_to_one",
    indicator=True,
)

print(combined.shape)
print(combined["_merge"].value_counts())
output_path = "C:/Users/tejas/Projects/amazon-customer-intelligence/data/amazon_electronics_combined.csv"

combined.to_csv(output_path, index=False)

print(f"\nSaved combined dataset to: {output_path}")
print(f"Combined dataset shape: {combined.shape}")
