#Data Quality Check
#imports
import numpy as np
import pandas as pd

#Load Dataset Function
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print("dataset loaded successfully")
        return df
    except FileNotFoundError:
        print("Error File not found ")
        return None

def inspect(df):
    print(f"-------First 5 rows------:\n{df.head()}")
    return None

def data_dimensions(df):
    print(f"-------Data Dimensions------:\n{df.shape}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

def data_types(df):
     print(f"-------Data Types------:\n{df.dtypes}")

def column_names(df):
    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

def data_summaries(df):
     print(f"------Data Summaries------:\n{df.describe()}")

def count_missing(df):
     print(f"-------Count Missing Values------:\n{df.isnull().sum()}")

def duplicate_rows(df):
     print(f"-------Data Duplicates------:\n{df.duplicated().sum()}")



def validate_ratings(df):
    ratings = pd.to_numeric(df["rating"], errors="coerce")

    invalid = ratings.isna() | ~ratings.between(1, 5)

    print("\n--- INVALID RATINGS ---")
    print(f"Invalid or missing ratings: {invalid.sum()}")


def ratings_frequency(df):
    print("\n--- RATING FREQUENCY ---")
    print(df["rating"].value_counts(dropna=False).sort_index())

def missing_text(df):
    for column in ["review_title", "review_text"]:
        blank = (
            df[column].isna()
            |   df[column].astype("string").str.strip().eq("").fillna(False)
        )
        print(f"{column}: {blank.sum()} missing or blank")


def unix_convert(df):
    dates = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        errors="coerce",
    )


    #idt this is right
    print("\n--- REVIEW DATE RANGE ---")
    print(f"Earliest: {dates.min()}")
    print(f"Latest:   {dates.max()}")
    print(f"Unparseable timestamps: {dates.isna().sum()}")

def verified_dist(df):
    print("\n--- VERIFIED PURCHASE DISTRIBUTION ---")
    print(df["verified_purchase"].value_counts(dropna=False))
    print("\nPercentages:")
    print(
        df["verified_purchase"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )


def helpful_votes_dist(df):
    print("\n--- HELPFUL VOTES ---")
    print(df["helpful_votes"].describe())
    print("\nVote counts:")
    print(df["helpful_votes"].value_counts().sort_index())


def sampling_fit(df):
    print("\n--- SAMPLE REPRESENTATIVENESS CHECK ---")
    print("Sample size:", len(df))
    print("\nRating proportions:")
    print(
        df["rating"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )
    print(
        "\nCaution: the first 5,000 source records are not "
        "automatically representative of all Electronics reviews."
    )


def run_audit(df):
    inspect(df)
    data_dimensions(df)
    column_names(df)
    data_types(df)
    data_summaries(df)
    count_missing(df)
    duplicate_rows(df)
    validate_ratings(df)
    ratings_frequency(df)
    missing_text(df)
    verified_dist(df)
    helpful_votes_dist(df)
    sampling_fit(df)
