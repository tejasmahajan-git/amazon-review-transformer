# Amazon Customer Intelligence

**Customer Experience Analytics | NLP | Sentiment Analysis | Python**

An end-to-end analytics project exploring Amazon Electronics customer reviews to identify customer satisfaction patterns, recurring product complaints, sentiment trends, and opportunities for customer experience improvement.

The project combines exploratory data analysis, data quality validation, review text preprocessing, lexicon-based sentiment analysis, and transformer-based sentiment classification using DistilBERT.

## Project Objectives

- Analyze customer ratings and review patterns.
- Assess data quality and prepare a reproducible analytical dataset.
- Compare verified and unverified purchase reviews.
- Identify highly rated products and products with potential customer experience concerns.
- Apply sentiment analysis to customer feedback using both a simple baseline and a pretrained transformer.
- Identify recurring complaint themes such as reliability, battery performance, compatibility, setup, and usability.
- Translate analytical findings into potential business recommendations.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Data analysis and workflow automation |
| Pandas | Data cleaning, transformation, and analysis |
| Matplotlib | Data visualization |
| Scikit-learn | Text feature extraction and recurring-term analysis |
| Hugging Face Transformers | Pretrained DistilBERT sentiment classification |
| PyTorch | Transformer model inference |
| Git | Version control |

## Project Workflow

### Phase 1 — Data Quality Audit

- Inspected dataset dimensions, columns, and data types.
- Examined missing values and duplicate records.
- Validated customer ratings and helpful-vote values.
- Analyzed verified purchase status and rating distributions.
- Checked timestamp formats and merge results.

### Phase 2 — Data Integration

- Combined customer review records with product metadata.
- Validated the merge between reviews and product information.
- Examined unmatched records and the availability of product attributes.

### Phase 3 — Data Cleaning and Preparation

- Validated ratings, timestamps, prices, and helpful votes.
- Examined missing and empty review text.
- Investigated exact and potential duplicate reviews.
- Converted timestamps into usable date fields.
- Grouped ratings into low, middle, and high categories.
- Created normalized review text while preserving original text.
- Exported a clean analytical dataset and a cleaning report.

### Phase 4 — Exploratory Data Analysis

- Analyzed rating distributions and average ratings.
- Examined review volume over time.
- Compared verified and unverified purchase reviews.
- Investigated helpful votes and highly voted reviews.
- Compared products with sufficient review counts.
- Examined potential sampling bias and documented initial findings.

### Phase 5 — Sentiment and Customer Experience Analysis

- Established a simple lexicon-based sentiment baseline.
- Compared predicted sentiment with customer star ratings.
- Extracted recurring positive and negative terms.
- Identified interpretable customer experience themes.
- Examined theme frequency and associated ratings.
- Exported example reviews for manual inspection.

### Phase 5.2 — Transformer-Based Sentiment Analysis

- Applied a pretrained DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2).
- Ran inference on a manageable, rating-stratified sample.
- Compared transformer predictions with a simple sentiment baseline.
- Exported disagreements for manual investigation.
- Prepared a manually labelled validation sample.
- Documented model limitations and the distinction between sentiment, ratings, and complaint categories.

## Dataset

The project uses Amazon Electronics review data and associated product metadata.

The combined dataset contains **5,000 sampled review records** and 17 columns before analytical cleaning.

Key fields include:

- `product_id` — Product identifier
- `rating` — Customer star rating
- `review_title` — Original review title
- `review_text` — Original review text
- `timestamp` — Original review timestamp
- `helpful_votes` — Number of helpful votes
- `verified_purchase` — Verified purchase indicator
- `product_title` — Product title
- `brand` — Product brand
- `price` — Product price, where available

Additional fields are created during data preparation, including normalized review text, review dates, date groupings, and rating categories.

### Data Limitations

- The analysis uses a 5,000-record sample taken from the source data, not a confirmed random sample of all Amazon Electronics reviews.
- Findings may be affected by sampling bias and should not automatically be generalized to all Amazon customers.
- Some product metadata, particularly prices, is missing.
- Customer ratings and written sentiment do not always agree.
- Product-level comparisons depend on having sufficient review observations.

## Sentiment Analysis Methodology

### 1. Lexicon-Based Baseline

A simple word-list approach assigns sentiment using predefined positive and negative terms. This provides a transparent baseline against which the transformer model can be compared.

### 2. DistilBERT

The transformer experiment uses:

`distilbert/distilbert-base-uncased-finetuned-sst-2-english`

This pretrained model predicts **binary positive or negative sentiment**. It does not predict star ratings or identify specific complaint categories.

Model confidence scores are not proof of correctness. Performance must be evaluated against manually labelled review text before making accuracy claims.

### 3. Manual Validation

A stratified sample of reviews across star ratings is exported for human labelling. Once the labels have been completed, the predictions can be evaluated using metrics such as:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Sentiment predictions are treated as one analytical signal alongside customer ratings and review themes, rather than as ground truth.

### 4. Complaint Theme Analysis

Complaint themes are analyzed separately from sentiment classification. Example categories include:

- Reliability and product failures
- Battery and charging
- Compatibility and connectivity
- Setup and usability
- Performance
- Price and value
- Delivery and packaging

A review may mention multiple themes and may express positive overall sentiment while describing a specific product issue.

## Repository Structure

```text
amazon-customer-intelligence/
│
├── data/
│   ├── amazon_electronics_product_metadata.csv
│   ├── amazon_electronics_reviews_sample.csv
│   ├── amazon_electronics_combined.csv
│   ├── amazon_electronics_clean.csv
│   └── phase3_cleaning_report.txt
│
├── src/
│   ├── audit.py
│   ├── merge.py
│   ├── data_prep.py
│   ├── main.py
│   ├── analysis.py
│   ├── sentiment_analysis.py
│   └── sentiment_distilbert.py
│
├── outputs/
│   ├── phase4/
│   └── phase5_2/
│
├── README.md
└── .gitignore
```

*This structure describes the intended project layout. Generated outputs and intermediate datasets may vary as development progresses.*

## Installation and Usage

### Prerequisites

- Python 3.10 or a compatible Python version
- Git
- An internet connection for the initial pretrained model download

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd amazon-customer-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Alternatively, in Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install pandas matplotlib scikit-learn transformers torch
```

### 4. Run the data audit and preparation workflow

```bash
python src/main.py
```

This runs the data audit and data preparation functions configured in the main workflow.

### 5. Run exploratory data analysis

```bash
python src/analysis.py
```

Review the generated tables, charts, and findings in `outputs/phase4/`.

### 6. Run baseline sentiment and theme analysis

```bash
python src/sentiment_analysis.py
```

### 7. Run DistilBERT sentiment analysis

```bash
python src/sentiment_distilbert.py
```

The first transformer run downloads the pretrained model. Inference time depends on hardware, review length, and batch size.

**Note:** Run the data preparation workflow before the analysis scripts so the clean analytical dataset exists.

## Business Value

This project demonstrates how unstructured customer feedback can be transformed into structured analytical outputs that support business decisions.

Potential applications include:

- Prioritizing recurring product complaints.
- Investigating products associated with lower customer ratings.
- Identifying customer experience issues that warrant further investigation.
- Combining quantitative ratings with qualitative review evidence.
- Supporting product quality, customer support, and product management decisions.

The analysis identifies potential signals for further investigation; it does not establish that a particular product defect or business cause exists without additional evidence.

## Current Status

The project is organized as a multi-phase analytical workflow covering data quality, integration, preparation, exploratory analysis, and sentiment analysis.

The transformer-based experiment is designed to compare model predictions with a simpler baseline and prepare a human-labelled validation sample. Quantitative model performance should only be reported after the manual labels have been completed and evaluated.

## Future Improvements

- Complete manual sentiment labelling and calculate evaluation metrics.
- Improve baseline handling of negation and context.
- Refine complaint-theme classification beyond keyword matching.
- Add product-level complaint prioritization.
- Build an interactive dashboard for business stakeholders.
- Develop recommendations supported by quantified findings and representative review examples.

## Author

Developed as a portfolio project focused on Business Analytics, customer intelligence, data analysis, and applied natural language processing.
