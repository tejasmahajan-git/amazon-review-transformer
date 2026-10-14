
"""
Phase 5: Sentiment Analysis and Review Themes
Project: Amazon Customer Intelligence

Input:
    data/amazon_electronics_clean.csv

Outputs:
    outputs/phase5/

This is an interpretable rule-based baseline, not a trained AI model.
Sentiment predictions and star ratings are kept separate.
"""

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "amazon_electronics_clean.csv"
OUTPUT_DIR = ROOT / "outputs" / "phase5"

TEXT_COLUMN = "review_text_clean"
MIN_THEME_EXAMPLES = 3


# ==================================================
# 5.1 LOAD DATA AND ESTABLISH A BASELINE
# ==================================================

# Transparent starter lexicon. This is intentionally simple
# and should be evaluated before drawing strong conclusions.
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome",
    "love", "loved", "perfect", "best", "easy", "fast",
    "reliable", "useful", "recommend", "happy", "works",
    "working", "nice", "value", "comfortable", "impressed",
    "smooth", "clear", "solid", "fantastic", "wonderful",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "poor", "worst",
    "hate", "hated", "broken", "defective", "failed",
    "failure", "slow", "issue", "issues", "problem",
    "problems", "disappointed", "disappointing", "refund",
    "return", "returned", "unreliable", "waste", "cheap",
    "difficult", "hard", "annoying", "damage", "damaged",
    "doesn't", "not", "never", "stopped", "error",
}


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run Phase 3 first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    if "rating" not in df.columns:
        raise ValueError("Required column 'rating' is missing.")

    if TEXT_COLUMN not in df.columns:
        if "review_text" not in df.columns:
            raise ValueError(
                f"Neither '{TEXT_COLUMN}' nor 'review_text' exists."
            )
        df[TEXT_COLUMN] = (
            df["review_text"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Empty reviews cannot receive a meaningful text-based prediction.
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

    print(f"Loaded {len(df):,} reviews.")

    return df


def predict_baseline_sentiment(text):
    """
    Count positive and negative lexicon matches.

    Returns:
        Positive, Negative, or Neutral

    Limitation:
        Word counts do not understand context, negation,
        sarcasm, or domain-specific meaning.
    """
    words = re.findall(r"[a-z']+", text.lower())

    positive_count = sum(word in POSITIVE_WORDS for word in words)
    negative_count = sum(word in NEGATIVE_WORDS for word in words)

    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    return "Neutral"


def run_sentiment_baseline(df):
    print("\n--- 5.1 SENTIMENT BASELINE ---")

    df["sentiment"] = df[TEXT_COLUMN].apply(
        lambda text: (
            predict_baseline_sentiment(text)
            if text.strip()
            else "Unclassified"
        )
    )

    summary = (
        df["sentiment"]
        .value_counts(dropna=False)
        .rename_axis("sentiment")
        .reset_index(name="review_count")
    )

    summary["percentage"] = (
        summary["review_count"] / len(df) * 100
        if len(df)
        else 0
    )

    summary.to_csv(
        OUTPUT_DIR / "sentiment_distribution.csv", index=False
    )

    print(summary.round(2).to_string(index=False))

    return df


# ==================================================
# 5.2 COMPARE SENTIMENT WITH STAR RATINGS
# ==================================================

def compare_sentiment_and_ratings(df):
    print("\n--- 5.2 SENTIMENT VS STAR RATINGS ---")

    data = df.dropna(subset=["rating"]).copy()

    # Rating groups are descriptive labels, not sentiment labels.
    data["rating_group"] = pd.cut(
        data["rating"],
        bins=[0, 2, 3, 5],
        labels=["Low (1-2)", "Middle (3)", "High (4-5)"],
        include_lowest=True,
    )

    table = pd.crosstab(
        data["rating_group"],
        data["sentiment"],
        normalize="index",
    ).mul(100).round(2)

    table.to_csv(
        OUTPUT_DIR / "sentiment_by_rating_group.csv"
    )

    print("Sentiment percentages within each rating group:")
    print(table)

    # Inspect cases where the text baseline and star rating disagree.
    # This is a disagreement review, not proof that either label is wrong.
    comparable = data[
        data["sentiment"].isin(["Positive", "Negative"])
    ].copy()

    comparable["rating_direction"] = comparable["rating"].map(
        lambda rating: "Positive" if rating >= 4
        else ("Negative" if rating <= 2 else "Middle")
    )

    disagreements = comparable[
        (
            (comparable["sentiment"] == "Positive")
            & (comparable["rating"] <= 2)
        )
        |
        (
            (comparable["sentiment"] == "Negative")
            & (comparable["rating"] >= 4)
        )
    ]

    columns = [
        column for column in [
            "product_id", "rating", "sentiment",
            "review_title", "review_text",
        ]
        if column in disagreements.columns
    ]

    disagreements[columns].head(100).to_csv(
        OUTPUT_DIR / "sentiment_rating_disagreements.csv",
        index=False,
    )

    print(f"Potential disagreements exported: {len(disagreements):,}")


# ==================================================
# 5.3 RECURRING POSITIVE AND NEGATIVE TERMS
# ==================================================

def analyze_recurring_terms(df):
    print("\n--- 5.3 RECURRING TERMS ---")

    for sentiment in ["Positive", "Negative"]:
        texts = df.loc[
            (df["sentiment"] == sentiment)
            & df[TEXT_COLUMN].str.strip().ne(""),
            TEXT_COLUMN,
        ]

        if texts.empty:
            print(f"No {sentiment.lower()} reviews available.")
            continue

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=100,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )

        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            print(f"Not enough usable text for {sentiment.lower()} terms.")
            continue

        frequencies = matrix.sum(axis=0).A1

        term_table = pd.DataFrame({
            "term": vectorizer.get_feature_names_out(),
            "frequency": frequencies,
        }).sort_values("frequency", ascending=False)

        term_table.head(30).to_csv(
            OUTPUT_DIR / f"top_{sentiment.lower()}_terms.csv",
            index=False,
        )

        print(f"\nTop {sentiment.lower()} terms:")
        print(term_table.head(15).to_string(index=False))


# ==================================================
# 5.4 GROUP REVIEWS INTO INTERPRETABLE THEMES
# ==================================================

THEMES = {
    "Reliability": {
        "reliable", "unreliable", "broken", "defective",
        "failed", "failure", "stopped", "durable",
        "durability", "working", "works",
    },
    "Battery": {
        "battery", "charging", "charge", "charged",
        "power", "drain", "draining", "runtime",
    },
    "Compatibility": {
        "compatible", "compatibility", "connect",
        "connection", "bluetooth", "pairing", "device",
        "devices", "android", "iphone", "windows",
    },
    "Setup and Usability": {
        "setup", "install", "installation", "configure",
        "configuration", "instructions", "manual",
        "easy", "difficult", "confusing", "interface",
    },
    "Performance": {
        "performance", "speed", "slow", "fast", "lag",
        "lagging", "responsive", "quality",
    },
    "Price and Value": {
        "price", "expensive", "cheap", "value", "cost",
        "worth", "overpriced", "refund",
    },
    "Delivery and Packaging": {
        "delivery", "shipping", "packaging", "package",
        "arrived", "damaged", "missing", "box",
    },
}


def assign_themes(text):
    """
    Assign all themes whose keyword sets match the review.
    A review may belong to multiple themes.
    """
    words = set(re.findall(r"[a-z']+", text.lower()))

    found = [
        theme for theme, keywords in THEMES.items()
        if words.intersection(keywords)
    ]

    return found if found else ["Other / Unclassified"]


def analyze_themes(df):
    print("\n--- 5.4 THEME CLASSIFICATION ---")

    df["themes"] = df[TEXT_COLUMN].apply(assign_themes)

    # Explode multi-theme reviews into one row per theme.
    exploded = df.explode("themes").copy()

    exploded["is_low_rating"] = exploded["rating"].isin([1, 2])

    theme_summary = (
        exploded.groupby("themes")
        .agg(
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
            low_rating_count=("is_low_rating", "sum"),
        )
        .reset_index()
        .rename(columns={"themes": "theme"})
    )

    theme_summary["percentage_of_reviews"] = (
        theme_summary["review_count"] / len(df) * 100
        if len(df)
        else 0
    )

    theme_summary["low_rating_percentage"] = (
        theme_summary["low_rating_count"]
        / theme_summary["review_count"] * 100
    )

    theme_summary = theme_summary.sort_values(
        "review_count", ascending=False
    )

    theme_summary.to_csv(
        OUTPUT_DIR / "theme_summary.csv", index=False
    )

    print(theme_summary.round(2).to_string(index=False))

    return df, exploded, theme_summary


# ==================================================
# 5.5 QUANTIFY THEME FREQUENCY AND RATINGS
# ==================================================

def quantify_themes(exploded):
    print("\n--- 5.5 THEME FREQUENCY AND RATINGS ---")

    # This is an explicit export of theme-level rating patterns.
    # A review assigned to multiple themes contributes to each theme.
    summary = (
        exploded.groupby("themes")
        .agg(
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
            median_rating=("rating", "median"),
            one_two_star_rate=(
                "rating",
                lambda s: s.isin([1, 2]).mean() * 100,
            ),
            four_five_star_rate=(
                "rating",
                lambda s: s.isin([4, 5]).mean() * 100,
            ),
        )
        .reset_index()
        .rename(columns={"themes": "theme"})
        .sort_values("review_count", ascending=False)
    )

    summary.to_csv(
        OUTPUT_DIR / "theme_rating_analysis.csv", index=False
    )

    print(summary.round(2).to_string(index=False))


# ==================================================
# 5.6 EXPORT EXAMPLES FOR MANUAL THEME REVIEW
# ==================================================

def export_theme_examples(df):
    print("\n--- 5.6 THEME EXAMPLES ---")

    records = []

    for _, row in df.iterrows():
        themes = row["themes"]

        for theme in themes:
            if theme == "Other / Unclassified":
                continue

            records.append({
                "product_id": row.get("product_id"),
                "rating": row.get("rating"),
                "sentiment": row.get("sentiment"),
                "theme": theme,
                "review_title": row.get("review_title"),
                "review_text": row.get("review_text"),
            })

    examples = pd.DataFrame(records)

    if examples.empty:
        print("No theme examples were found.")
        return

    # Keep a small sample from each theme for human inspection.
    examples.groupby("theme", group_keys=False).head(10).to_csv(
        OUTPUT_DIR / "theme_review_examples.csv", index=False
    )

    print("Saved theme_review_examples.csv for manual verification.")


# ==================================================
# 5.7 EVALUATE LIMITATIONS AND MISCLASSIFICATIONS
# ==================================================

def write_evaluation_notes(df):
    print("\n--- 5.7 EVALUATION AND LIMITATIONS ---")

    unclassified = (df["sentiment"] == "Unclassified").sum()
    neutral = (df["sentiment"] == "Neutral").sum()

    notes = [
        "PHASE 5 BASELINE EVALUATION NOTES",
        "=" * 40,
        "",
        "Method",
        "- Sentiment uses a small, manually defined word lexicon.",
        "- Theme assignment uses predefined keyword lists.",
        "- A review may be assigned to more than one theme.",
        "",
        "Current coverage",
        f"- Total reviews: {len(df):,}",
        f"- Empty/unclassified reviews: {unclassified:,}",
        f"- Neutral baseline predictions: {neutral:,}",
        "",
        "Known limitations",
        "- The lexicon does not understand context or sarcasm.",
        "- Negation such as 'not good' may be misclassified.",
        "- Words such as 'cheap' can be positive or negative by context.",
        "- Theme keywords can miss synonyms and indirect complaints.",
        "- A keyword match is not proof that a theme is the true cause.",
        "- Star ratings are not ground-truth sentiment labels.",
        "",
        "Recommended manual evaluation",
        "1. Randomly inspect examples from each sentiment class.",
        "2. Manually label a small, diverse set of reviews.",
        "3. Compare the baseline against those human labels.",
        "4. Record false positives, false negatives, and ambiguous cases.",
        "5. Expand or revise the lexicon and theme rules based on evidence.",
        "",
        "No accuracy, precision, recall, or F1 score is reported because",
        "no independent human-labelled evaluation set is supplied.",
    ]

    (OUTPUT_DIR / "baseline_evaluation_notes.txt").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )

    print("\n".join(notes))


# ==================================================
# 5.8 KEEP SENTIMENT SEPARATE FROM STAR RATINGS
# ==================================================

def save_analytical_dataset(df):
    print("\n--- 5.8 SEPARATE SENTIMENT AND RATINGS ---")

    output_columns = list(df.columns)

    # Both columns are preserved independently.
    # No sentiment labels are inferred from the star rating.
    output_file = OUTPUT_DIR / "reviews_with_sentiment_and_themes.csv"
    df.to_csv(output_file, index=False)

    print("Preserved separate columns: rating, sentiment, themes")
    print(f"Saved enriched review dataset to:\n{output_file}")


# ==================================================
# MAIN WORKFLOW
# ==================================================

def main():
    df = load_data()

    df = run_sentiment_baseline(df)
    compare_sentiment_and_ratings(df)
    analyze_recurring_terms(df)

    df, exploded, theme_summary = analyze_themes(df)
    quantify_themes(exploded)
    export_theme_examples(df)
    write_evaluation_notes(df)
    save_analytical_dataset(df)

    print("\n" + "=" * 50)
    print("PHASE 5 COMPLETED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
