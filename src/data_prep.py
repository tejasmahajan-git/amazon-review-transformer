
"""
Phase 3: Data Preparation and Cleaning
Project: Amazon Customer Intelligence

Input:
    data/amazon_electronics_combined.csv

Outputs:
    data/amazon_electronics_clean.csv
    data/phase3_cleaning_report.txt

The original input dataset is never overwritten.
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = ROOT / "data" / "amazon_electronics_clean.csv"
REPORT_FILE = ROOT / "data" / "phase3_cleaning_report.txt"


# --------------------------------------------------
# 3.1 CHECK INVALID VALUES AND FORMATS
# --------------------------------------------------

def validate_data(df):
    print("\n--- 3.1 DATA VALIDATION ---")

    required_columns = [
        "product_id",
        "rating",
        "timestamp",
        "helpful_votes",
        "verified_purchase",
        "review_title",
        "review_text",
        "price",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    ratings = pd.to_numeric(df["rating"], errors="coerce")
    votes = pd.to_numeric(df["helpful_votes"], errors="coerce")
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
    prices = pd.to_numeric(df["price"], errors="coerce")

    dates = pd.to_datetime(
        timestamps, unit="ms", errors="coerce"
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing prices: {prices.isna().sum():,}")
    print(f"Invalid or missing ratings: "
          f"{(ratings.isna() | ~ratings.between(1, 5)).sum():,}")
    print(f"Missing/non-numeric helpful votes: "
          f"{votes.isna().sum():,}")
    print(f"Negative helpful votes: {(votes < 0).sum():,}")
    print(f"Missing/non-numeric timestamps: "
          f"{timestamps.isna().sum():,}")
    print(f"Unconvertible timestamps: {dates.isna().sum():,}")
    print(f"Negative prices: {(prices < 0).sum():,}")

    # Report unexpected values without silently changing the source.
    verified_values = df["verified_purchase"].dropna().unique()
    print(f"Verified-purchase values: {verified_values}")


# --------------------------------------------------
# 3.2 INSPECT AND HANDLE EMPTY REVIEW TEXT
# --------------------------------------------------

def inspect_review_text(df):
    print("\n--- 3.2 REVIEW TEXT QUALITY ---")

    for column in ["review_title", "review_text"]:
        text = df[column].astype("string")

        empty = (
            text.isna()
            | text.str.strip().eq("").fillna(False)
        )

        print(f"{column}: {empty.sum():,} missing or blank")


# --------------------------------------------------
# 3.3 CHECK DUPLICATE REVIEWS
# --------------------------------------------------

def inspect_duplicates(df):
    print("\n--- 3.3 DUPLICATE REVIEW CHECK ---")

    exact_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_duplicates:,}")

    # Compare reviews using available identifying fields.
    # Empty review text is excluded to avoid false matches.
    title = (
        df["review_title"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    text = (
        df["review_text"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    candidate = pd.DataFrame({
        "product_id": df["product_id"],
        "rating": pd.to_numeric(df["rating"], errors="coerce"),
        "title": title,
        "text": text,
    })

    eligible = candidate["text"].ne("")

    duplicate_candidates = candidate.loc[eligible].duplicated(
        subset=["product_id", "rating", "title", "text"]
    ).sum()

    print(
        "Potential duplicate reviews beyond first occurrence: "
        f"{duplicate_candidates:,}"
    )

    print(
        "Limitation: matching text and product does not prove "
        "that two records are the same review. Different customers "
        "may write identical text."
    )

    print("No rows are deleted by this check.")


# --------------------------------------------------
# 3.4 CONVERT TIMESTAMPS AND CREATE DATE FIELDS
# --------------------------------------------------

def create_date_fields(df):
    print("\n--- 3.4 DATE FIELDS ---")

    timestamps = pd.to_numeric(
        df["timestamp"], errors="coerce"
    )

    dates = pd.to_datetime(
        timestamps, unit="ms", errors="coerce", utc=True
    )

    # Convert to timezone-naive UTC for simpler CSV analysis.
    df["review_date"] = dates.dt.tz_convert(None)
    df["review_year"] = dates.dt.year
    df["review_month"] = dates.dt.month
    df["review_year_month"] = dates.dt.strftime("%Y-%m")

    print(f"Dates created: {df['review_date'].notna().sum():,}")
    print(f"Missing dates: {df['review_date'].isna().sum():,}")


# --------------------------------------------------
# 3.5 CREATE RATING GROUPS
# --------------------------------------------------

def create_rating_groups(df):
    print("\n--- 3.5 RATING GROUPS ---")

    rating = pd.to_numeric(df["rating"], errors="coerce")

    df["rating_group"] = pd.cut(
        rating,
        bins=[0, 2, 3, 5],
        labels=["Low", "Middle", "High"],
        include_lowest=True,
    )

    print(df["rating_group"].value_counts(dropna=False))


# --------------------------------------------------
# 3.6 NORMALIZE TEXT, PRESERVE ORIGINALS
# --------------------------------------------------

def normalize_review_text(df):
    print("\n--- 3.6 TEXT NORMALIZATION ---")

    for original, cleaned in [
        ("review_title", "review_title_clean"),
        ("review_text", "review_text_clean"),
    ]:
        # Keep the original column unchanged.
        df[cleaned] = (
            df[original]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.lower()
        )

    print("Created review_title_clean and review_text_clean.")
    print("Original review title and text have been preserved.")


# --------------------------------------------------
# 3.7 + 3.8 REPRODUCIBLE CLEANING AND SAVING
# --------------------------------------------------

def save_clean_dataset(df):
    print("\n--- 3.7 / 3.8 SAVE CLEAN DATASET ---")

    # Remove the merge diagnostic column from the analytical
    # output; the merge was already checked in the audit/merge step.
    if "_merge" in df.columns:
        df = df.drop(columns=["_merge"])

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Clean dataset saved to:\n{OUTPUT_FILE}")
    print(f"Output shape: {df.shape}")

    return df


def save_cleaning_report(original_df, clean_df):
    report = [
        "AMAZON CUSTOMER INTELLIGENCE",
        "PHASE 3: DATA CLEANING REPORT",
        "=" * 45,
        "",
        f"Original row count: {len(original_df):,}",
        f"Clean dataset row count: {len(clean_df):,}",
        f"Original column count: {len(original_df.columns)}",
        f"Clean dataset column count: {len(clean_df.columns)}",
        "",
        "CLEANING DECISIONS",
        "-" * 45,
        "1. Original source dataset was not overwritten.",
        "2. Rows with missing prices were preserved.",
        "3. Rows with empty review text were preserved.",
        "4. Exact and potential duplicates were inspected,",
        "   but no rows were removed automatically.",
        "5. Unix millisecond timestamps were converted to dates.",
        "6. Rating groups: Low (1-2), Middle (3), High (4-5).",
        "7. Lowercase, trimmed text was saved in separate columns.",
        "8. Original review title and text were preserved.",
        "9. The _merge diagnostic column was excluded from output.",
        "",
        "LIMITATIONS",
        "-" * 45,
        "The sample contains the first 5,000 source records and",
        "may not represent all Amazon Electronics reviews.",
        "Matching review text does not prove duplicate identity.",
        "Missing prices may limit price-related analyses.",
        "Text normalization does not identify review themes",
        "or determine sentiment.",
        "",
    ]

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"Cleaning report saved to:\n{REPORT_FILE}")


# --------------------------------------------------
# 3.9 VALIDATE THE CLEAN DATASET
# --------------------------------------------------

def validate_clean_dataset(original_df, clean_df):
    print("\n--- 3.9 POST-CLEANING VALIDATION ---")

    row_count_matches = len(original_df) == len(clean_df)

    print(f"Original rows: {len(original_df):,}")
    print(f"Clean rows: {len(clean_df):,}")
    print(f"Row count preserved: {row_count_matches}")

    required_output_columns = [
        "product_id",
        "rating",
        "review_text",
        "review_text_clean",
        "timestamp",
        "review_date",
        "rating_group",
    ]

    missing_columns = [
        column for column in required_output_columns
        if column not in clean_df.columns
    ]

    print(f"Missing required output columns: {missing_columns}")

    ratings = pd.to_numeric(clean_df["rating"], errors="coerce")
    invalid_ratings = ratings.isna() | ~ratings.between(1, 5)

    print(f"Invalid ratings after preparation: "
          f"{invalid_ratings.sum():,}")
    print(f"Missing product IDs: {clean_df['product_id'].isna().sum():,}")
    print(f"Missing review dates: {clean_df['review_date'].isna().sum():,}")

    if not row_count_matches:
        raise ValueError("Row count changed unexpectedly.")

    if missing_columns:
        raise ValueError(
            f"Required output columns missing: {missing_columns}"
        )

    print("Validation completed.")


# --------------------------------------------------
# MAIN PHASE 3 WORKFLOW
# --------------------------------------------------

def run_data_prep(df):
    print("\n" + "=" * 45)
    print("STARTING PHASE 3: DATA PREPARATION")
    print("=" * 45)

    original_df = df.copy()

    # Inspect and report data quality.
    validate_data(original_df)
    inspect_review_text(original_df)
    inspect_duplicates(original_df)

    # Prepare a separate analytical copy.
    clean_df = original_df.copy()

    create_date_fields(clean_df)
    create_rating_groups(clean_df)
    normalize_review_text(clean_df)

    # Save and validate outputs.
    clean_df = save_clean_dataset(clean_df)
    save_cleaning_report(original_df, clean_df)
    validate_clean_dataset(original_df, clean_df)

    print("\nPHASE 3 COMPLETED")
