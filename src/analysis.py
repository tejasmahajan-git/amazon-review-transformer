
"""
Phase 4: Exploratory Data Analysis (EDA)
Project: Amazon Customer Intelligence

Input:
    data/amazon_electronics_clean.csv

Outputs:
    outputs/phase4/  - CSV tables, charts, and findings report

The script does not modify the input dataset.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "amazon_electronics_clean.csv"
OUTPUT_DIR = ROOT / "outputs" / "phase4"

# Products need at least this many sampled reviews
# before being included in product comparisons.
MIN_PRODUCT_REVIEWS = 20


def setup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {INPUT_FILE}\n"
            "Run the Phase 3 cleaning script first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "product_id",
        "rating",
        "timestamp",
        "helpful_votes",
        "verified_purchase",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # Ensure numerical columns have appropriate types.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_votes"] = pd.to_numeric(
        df["helpful_votes"], errors="coerce"
    )

    # The source timestamps are Unix milliseconds.
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(
            df["review_date"], errors="coerce", utc=True
        )
    else:
        timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
        df["review_date"] = pd.to_datetime(
            timestamp, unit="ms", errors="coerce", utc=True
        )

    # Remove timezone information for consistent monthly grouping.
    df["review_date"] = df["review_date"].dt.tz_convert(None)

    # Ignore invalid ratings in rating-based calculations.
    df.loc[~df["rating"].between(1, 5), "rating"] = float("nan")

    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Output directory: {OUTPUT_DIR}")

    return df


# ==================================================
# 4.1 RATING COUNTS, PERCENTAGES, AND AVERAGE
# ==================================================

def analyze_ratings(df):
    print("\n--- 4.1 RATING ANALYSIS ---")

    ratings = df["rating"].dropna()

    counts = (
        ratings.value_counts()
        .reindex([1, 2, 3, 4, 5], fill_value=0)
        .rename_axis("rating")
        .reset_index(name="review_count")
    )

    counts["percentage"] = (
        counts["review_count"] / len(ratings) * 100
        if len(ratings)
        else 0
    )

    average_rating = ratings.mean()
    negative_rate = ratings.isin([1, 2]).mean() * 100 if len(ratings) else 0

    counts.to_csv(OUTPUT_DIR / "rating_distribution.csv", index=False)

    print(counts.to_string(index=False))
    print(f"\nAverage rating: {average_rating:.2f}/5")
    print(f"1-2 star review rate: {negative_rate:.2f}%")

    return counts, average_rating, negative_rate


# ==================================================
# 4.2 VISUALIZE RATING DISTRIBUTION
# ==================================================

def plot_rating_distribution(rating_table):
    print("\n--- 4.2 RATING DISTRIBUTION CHART ---")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        rating_table["rating"].astype(str),
        rating_table["review_count"],
    )

    ax.set_title("Amazon Electronics Review Rating Distribution")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rating_distribution.png", dpi=200)
    plt.close(fig)

    print("Saved rating_distribution.png")


# ==================================================
# 4.3 REVIEW VOLUME OVER TIME
# ==================================================

def analyze_review_volume(df):
    print("\n--- 4.3 REVIEW VOLUME OVER TIME ---")

    dated = df.dropna(subset=["review_date"]).copy()

    dated["review_month"] = (
        dated["review_date"].dt.to_period("M").astype(str)
    )

    monthly = (
        dated.groupby("review_month")
        .size()
        .reset_index(name="review_count")
    )

    monthly.to_csv(OUTPUT_DIR / "monthly_review_volume.csv", index=False)

    print(f"Rows with valid review dates: {len(dated):,}")
    print(f"Months represented: {len(monthly):,}")

    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(
            monthly["review_month"],
            monthly["review_count"],
            marker="o",
        )

        ax.set_title("Review Volume Over Time")
        ax.set_xlabel("Review Month")
        ax.set_ylabel("Number of Reviews")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "monthly_review_volume.png", dpi=200)
        plt.close(fig)

        print("Saved monthly_review_volume.png")

    return monthly


# ==================================================
# 4.4 VERIFIED VS UNVERIFIED PURCHASES
# ==================================================

def analyze_verified_purchases(df):
    print("\n--- 4.4 VERIFIED PURCHASE ANALYSIS ---")

    data = df.dropna(subset=["rating", "verified_purchase"]).copy()

    # Standardize common Boolean/string representations.
    values = (
        data["verified_purchase"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    data["purchase_status"] = values.map({
        "true": "Verified",
        "false": "Unverified",
        "1": "Verified",
        "0": "Unverified",
        "yes": "Verified",
        "no": "Unverified",
    })

    data = data.dropna(subset=["purchase_status"])

    summary = (
        data.groupby("purchase_status")
        .agg(
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
            one_two_star_reviews=(
                "rating",
                lambda s: s.isin([1, 2]).sum(),
            ),
        )
        .reset_index()
    )

    summary["negative_review_percentage"] = (
        summary["one_two_star_reviews"] / summary["review_count"] * 100
    )

    summary.to_csv(
        OUTPUT_DIR / "verified_purchase_comparison.csv", index=False
    )

    print(summary.round(2).to_string(index=False))

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            summary["purchase_status"],
            summary["average_rating"],
        )

        ax.set_title("Average Rating: Verified vs Unverified Purchases")
        ax.set_xlabel("Purchase Status")
        ax.set_ylabel("Average Rating (out of 5)")
        ax.set_ylim(0, 5)
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "verified_purchase_ratings.png", dpi=200)
        plt.close(fig)

        print("Saved verified_purchase_ratings.png")

    return summary


# ==================================================
# 4.5 HELPFUL VOTES AND HIGHLY VOTED REVIEWS
# ==================================================

def analyze_helpful_votes(df):
    print("\n--- 4.5 HELPFUL VOTES ---")

    votes = df["helpful_votes"].dropna()

    if votes.empty:
        print("No valid helpful-vote values available.")
        return pd.DataFrame()

    print(f"Average helpful votes: {votes.mean():.2f}")
    print(f"Median helpful votes: {votes.median():.0f}")
    print(f"Maximum helpful votes: {votes.max():.0f}")
    print(f"Reviews with zero votes: {(votes == 0).sum():,}")
    print(f"Reviews with at least one vote: {(votes > 0).sum():,}")

    # Helpful votes are usually skewed, so use a log scale.
    fig, ax = plt.subplots(figsize=(8, 5))

    positive_votes = votes[votes > 0]

    if not positive_votes.empty:
        ax.hist(positive_votes, bins=30)
        ax.set_xscale("log")
        ax.set_title("Distribution of Positive Helpful-Vote Counts")
        ax.set_xlabel("Helpful Votes (log scale)")
        ax.set_ylabel("Number of Reviews")
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "helpful_votes_distribution.png", dpi=200)

    plt.close(fig)

    # Export highly voted reviews for manual inspection.
    review_columns = [
        column for column in [
            "product_id",
            "product_title",
            "rating",
            "review_title",
            "review_text",
            "helpful_votes",
            "verified_purchase",
            "review_date",
        ]
        if column in df.columns
    ]

    top_reviews = (
        df.dropna(subset=["helpful_votes"])
        .sort_values("helpful_votes", ascending=False)
        [review_columns]
        .head(20)
    )

    top_reviews.to_csv(
        OUTPUT_DIR / "top_20_helpful_reviews.csv", index=False
    )

    print("Saved top_20_helpful_reviews.csv")
    print("Saved helpful_votes_distribution.png")

    return top_reviews


# ==================================================
# 4.6 PRODUCT-LEVEL COMPARISONS
# ==================================================

def analyze_products(df):
    print("\n--- 4.6 PRODUCT COMPARISON ---")

    data = df.dropna(subset=["product_id", "rating"]).copy()

    product_summary = (
        data.groupby("product_id")
        .agg(
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
            low_rating_count=(
                "rating",
                lambda s: s.isin([1, 2]).sum(),
            ),
        )
        .reset_index()
    )

    product_summary["low_rating_percentage"] = (
        product_summary["low_rating_count"]
        / product_summary["review_count"] * 100
    )

    if "product_title" in data.columns:
        titles = (
            data.dropna(subset=["product_title"])
            .groupby("product_id")["product_title"]
            .first()
            .rename("product_title")
        )

        product_summary = product_summary.merge(
            titles, on="product_id", how="left"
        )

    eligible = product_summary[
        product_summary["review_count"] >= MIN_PRODUCT_REVIEWS
    ].copy()

    eligible = eligible.sort_values(
        ["average_rating", "review_count"],
        ascending=[False, False],
    )

    eligible.to_csv(
        OUTPUT_DIR / "product_comparison.csv", index=False
    )

    print(f"Unique products: {len(product_summary):,}")
    print(
        f"Products with at least {MIN_PRODUCT_REVIEWS} sampled reviews: "
        f"{len(eligible):,}"
    )

    if not eligible.empty:
        display_columns = [
            column for column in [
                "product_id",
                "product_title",
                "review_count",
                "average_rating",
                "low_rating_percentage",
            ]
            if column in eligible.columns
        ]

        print("\nHighest-rated eligible products:")
        print(eligible[display_columns].head(10).round(2).to_string(index=False))

        worst = eligible.sort_values(
            ["average_rating", "review_count"],
            ascending=[True, False],
        )

        print("\nLowest-rated eligible products:")
        print(worst[display_columns].head(10).round(2).to_string(index=False))

    print(
        "Caution: these rankings use the sampled reviews, not each "
        "product's full Amazon rating history."
    )

    return eligible


# ==================================================
# 4.7 INVESTIGATE POSSIBLE SAMPLING BIAS
# ==================================================

def analyze_sampling_bias(df):
    print("\n--- 4.7 SAMPLING-BIAS CHECKS ---")

    findings = []

    findings.append(
        "The dataset contains the first 5,000 source records. "
        "Record order may affect which reviews are included."
    )

    findings.append(
        "The results describe this sample and should not automatically "
        "be generalized to all Amazon Electronics reviews."
    )

    # Check whether a small number of products dominate the sample.
    product_counts = df["product_id"].value_counts(dropna=True)

    if len(df):
        top_five_share = product_counts.head(5).sum() / len(df) * 100
    else:
        top_five_share = 0

    print(f"Unique products in sample: {df['product_id'].nunique():,}")
    print(f"Share of all rows from top 5 products: {top_five_share:.2f}%")

    findings.append(
        f"The five most-represented products account for "
        f"{top_five_share:.2f}% of all rows. Product concentration may "
        "cause the overall results to reflect a small number of products."
    )

    # Compare the first and second halves in their existing source order.
    midpoint = len(df) // 2

    if midpoint > 0 and len(df) - midpoint > 0:
        first_half = df.iloc[:midpoint]
        second_half = df.iloc[midpoint:]

        first_mean = first_half["rating"].mean()
        second_mean = second_half["rating"].mean()

        first_date = first_half["review_date"].median()
        second_date = second_half["review_date"].median()

        print(f"First-half mean rating: {first_mean:.2f}")
        print(f"Second-half mean rating: {second_mean:.2f}")

        findings.append(
            f"The first half has an average rating of {first_mean:.2f}; "
            f"the second half has {second_mean:.2f}. A difference may "
            "indicate ordering effects, but does not prove sampling bias."
        )

        findings.append(
            f"Median review dates by source order: first half = "
            f"{first_date}, second half = {second_date}. If these differ "
            "substantially, the source order may also be time-related."
        )

    findings.append(
        "A first-record sample cannot establish population "
        "representativeness without knowing the source ordering and "
        "the population sampling process."
    )

    for finding in findings:
        print("- " + finding)

    (OUTPUT_DIR / "sampling_bias_notes.txt").write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )

    return findings


# ==================================================
# 4.8 EVIDENCE-BACKED INITIAL FINDINGS
# ==================================================

def write_initial_findings(
    df,
    rating_table,
    average_rating,
    negative_rate,
    monthly,
    verified,
    product_summary,
    sampling_findings,
):
    print("\n--- 4.8 INITIAL FINDINGS ---")

    lines = [
        "AMAZON CUSTOMER INTELLIGENCE",
        "PHASE 4: INITIAL EDA FINDINGS",
        "=" * 50,
        "",
        "Dataset scope",
        f"- Reviews analyzed: {len(df):,}",
        f"- Products represented: {df['product_id'].nunique():,}",
        "- The sample consists of the first 5,000 source records.",
        "- Findings describe the available sample, not all Amazon customers.",
        "",
        "Rating distribution",
        f"- Average rating: {average_rating:.2f}/5.",
        f"- Reviews rated 1 or 2 stars: {negative_rate:.2f}%.",
    ]

    five_star = rating_table.loc[rating_table["rating"] == 5]
    if not five_star.empty:
        lines.append(
            f"- Five-star reviews represent "
            f"{five_star.iloc[0]['percentage']:.2f}% of valid ratings."
        )

    lines.extend(["", "Review volume over time"])

    if not monthly.empty:
        busiest = monthly.loc[monthly["review_count"].idxmax()]
        lines.append(
            f"- The highest-volume month in this sample is "
            f"{busiest['review_month']} "
            f"({int(busiest['review_count']):,} reviews)."
        )
        lines.append(
            "- Monthly counts describe the sample's recorded dates; "
            "they do not necessarily measure actual platform-wide volume."
        )
    else:
        lines.append("- No valid dates were available for monthly analysis.")

    lines.extend(["", "Verified purchase comparison"])

    if not verified.empty:
        for _, row in verified.iterrows():
            lines.append(
                f"- {row['purchase_status']}: {int(row['review_count']):,} "
                f"reviews; average rating {row['average_rating']:.2f}; "
                f"1-2 star rate {row['negative_review_percentage']:.2f}%."
            )
        lines.append(
            "- Differences are descriptive associations, not evidence "
            "that verification status causes a rating difference."
        )
    else:
        lines.append("- No usable verified-purchase groups were available.")

    lines.extend(["", "Product comparisons"])

    if not product_summary.empty:
        lines.append(
            f"- {len(product_summary):,} products met the minimum "
            f"threshold of {MIN_PRODUCT_REVIEWS} sampled reviews."
        )
        lines.append(
            "- Product comparisons are based only on the sampled records."
        )
    else:
        lines.append(
            f"- No products met the threshold of "
            f"{MIN_PRODUCT_REVIEWS} sampled reviews."
        )

    lines.extend(["", "Sampling limitations"])

    for finding in sampling_findings:
        lines.append("- " + finding)

    lines.extend([
        "",
        "Interpretation cautions",
        "- Rating and review patterns do not establish causation.",
        "- Helpful votes measure observed engagement, not necessarily quality.",
        "- These initial summaries do not identify the causes of dissatisfaction.",
        "- Further text/theme analysis is needed before recommending specific fixes.",
        "",
    ])

    report = "\n".join(lines)
    (OUTPUT_DIR / "initial_findings.txt").write_text(
        report, encoding="utf-8"
    )

    print(report)
    print(f"\nSaved findings to {OUTPUT_DIR / 'initial_findings.txt'}")


# ==================================================
# RUN PHASE 4
# ==================================================

def main():
    df = setup()

    rating_table, average_rating, negative_rate = analyze_ratings(df)
    plot_rating_distribution(rating_table)

    monthly = analyze_review_volume(df)
    verified = analyze_verified_purchases(df)
    analyze_helpful_votes(df)
    products = analyze_products(df)
    sampling = analyze_sampling_bias(df)

    write_initial_findings(
        df=df,
        rating_table=rating_table,
        average_rating=average_rating,
        negative_rate=negative_rate,
        monthly=monthly,
        verified=verified,
        product_summary=products,
        sampling_findings=sampling,
    )

    print("\n" + "=" * 50)
    print("PHASE 4 COMPLETED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
