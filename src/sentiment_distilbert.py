
from pathlib import Path
import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "amazon_electronics_clean.csv"
OUTPUT_DIR = ROOT / "outputs" / "phase5_2"

MODEL_NAME = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
SAMPLE_SIZE = 300
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(INPUT_FILE)

    if "rating" not in df.columns:
        raise ValueError("The dataset must contain a rating column.")

    if "review_text_clean" in df.columns:
        text_col = "review_text_clean"
    elif "review_text" in df.columns:
        text_col = "review_text"
    else:
        raise ValueError("No review text column was found.")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["analysis_text"] = df[text_col].fillna("").astype(str).str.strip()

    # Empty reviews cannot be classified meaningfully.
    df = df[df["analysis_text"].ne("")].copy()

    # Stratify by star rating so the sample includes different ratings.
    df["rating"] = df["rating"].astype(int)
    sample = (
        df.groupby("rating", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(
                    len(group),
                    max(1, round(SAMPLE_SIZE * len(group) / len(df)))
                ),
                random_state=RANDOM_STATE
            ),
            include_groups=True
        )
    )

    # Keep the experiment manageable.
    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

    return sample.reset_index(drop=True)


def run_distilbert(df):
    print(f"Loading pretrained model: {MODEL_NAME}")

    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1
    )

    texts = df["analysis_text"].tolist()

    # Truncate long reviews to the model's supported input length.
    predictions = classifier(
        texts,
        batch_size=8,
        truncation=True,
        max_length=512
    )

    df["distilbert_sentiment"] = [
        prediction["label"].capitalize()
        for prediction in predictions
    ]
    df["distilbert_confidence"] = [
        prediction["score"]
        for prediction in predictions
    ]

    return df


def compare_with_baseline(df):
    # The earlier Phase 5 script uses Positive / Negative / Neutral.
    baseline_file = OUTPUT_DIR / "baseline_predictions.csv"

    if baseline_file.exists():
        baseline = pd.read_csv(baseline_file)

        if "baseline_sentiment" in baseline.columns:
            df = df.merge(
                baseline[["baseline_sentiment"]],
                left_index=True,
                right_index=True,
                how="left"
            )

    # If the earlier script did not export a compatible baseline,
    # use a clearly labelled simple word-list baseline here.
    if "baseline_sentiment" not in df.columns:
        positive_words = {
            "great", "excellent", "love", "good", "perfect",
            "amazing", "best", "easy", "works", "recommend"
        }
        negative_words = {
            "bad", "terrible", "poor", "hate", "broken",
            "waste", "defective", "disappointed", "worst", "failed"
        }

        def baseline_label(text):
            words = set(text.lower().split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)

            if pos > neg:
                return "Positive"
            if neg > pos:
                return "Negative"
            return "Neutral"

        df["baseline_sentiment"] = df["analysis_text"].apply(
            baseline_label
        )

    df["sentiment_disagreement"] = (
        df["distilbert_sentiment"] != df["baseline_sentiment"]
    )

    df["rating_group"] = pd.cut(
        df["rating"],
        bins=[0, 2, 3, 5],
        labels=["Low (1-2)", "Middle (3)", "High (4-5)"]
    )

    return df


def export_results(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_DIR / "distilbert_predictions.csv",
        index=False
    )

    disagreements = df[df["sentiment_disagreement"]].copy()
    disagreements.to_csv(
        OUTPUT_DIR / "sentiment_disagreements.csv",
        index=False
    )

    comparison = pd.crosstab(
        df["baseline_sentiment"],
        df["distilbert_sentiment"],
        normalize="index"
    ).mul(100).round(2)

    comparison.to_csv(
        OUTPUT_DIR / "baseline_vs_distilbert_percent.csv"
    )

    rating_comparison = pd.crosstab(
        df["rating_group"],
        df["distilbert_sentiment"],
        normalize="index"
    ).mul(100).round(2)

    rating_comparison.to_csv(
        OUTPUT_DIR / "distilbert_by_star_rating_percent.csv"
    )

    # A balanced manual-review worksheet:
    # up to 20 records from each rating, where available.
    manual_sample = (
        df.groupby("rating", group_keys=False, observed=True)
        .apply(
            lambda group: group.sample(
                n=min(20, len(group)),
                random_state=RANDOM_STATE
            )
        )
        .copy()
    )

    manual_sample["human_sentiment"] = ""
    manual_sample["reviewer_notes"] = ""

    manual_sample.to_csv(
        OUTPUT_DIR / "manual_validation_sample.csv",
        index=False
    )

    notes = """
DISTILBERT SENTIMENT EXPERIMENT

Model:
distilbert/distilbert-base-uncased-finetuned-sst-2-english

Task:
Binary positive/negative sentiment classification, fine-tuned on SST-2.

Important limitations:
- This model does not predict 1-5 star ratings.
- It does not classify complaint themes such as battery or reliability.
- Confidence scores are model scores, not proof that a prediction is correct.
- Product reviews may differ from the model's training domain.
- Long reviews may be truncated to fit the model input limit.
- Agreement with the word-list baseline does not establish correctness.
- The sample is stratified by rating and is not a random sample of all
  Amazon Electronics reviews.

Validation:
Fill in human_sentiment in manual_validation_sample.csv using
Positive or Negative after reading each review. Use Unclear if needed.
Compare predictions with these human labels before making accuracy claims.
"""
    (OUTPUT_DIR / "model_limitations.txt").write_text(
        notes.strip(),
        encoding="utf-8"
    )

    print("\n--- DISTILBERT EXPERIMENT COMPLETE ---")
    print(f"Reviews analyzed: {len(df)}")
    print(
        "Baseline/model disagreements:",
        int(df["sentiment_disagreement"].sum())
    )
    print(
        "Disagreement percentage:",
        round(100 * df["sentiment_disagreement"].mean(), 2),
        "%"
    )
    print(f"Results saved to: {OUTPUT_DIR}")


def main():
    df = load_data()
    print(f"Stratified sample size: {len(df)}")

    df = run_distilbert(df)
    df = compare_with_baseline(df)
    export_results(df)


if __name__ == "__main__":
    main()
