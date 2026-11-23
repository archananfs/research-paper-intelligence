
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from pathlib import Path


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "arxiv_classification_dataset.csv"
)

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. BASIC DATA QUALITY
# --------------------------------------------------

print("\n--- DATA QUALITY ---")

print("Missing values:")
print(df.isnull().sum())

print("\nEmpty abstracts:")
print((df["abstract"].fillna("").str.strip() == "").sum())

print("\nDuplicate paper IDs:")
print(df["paper_id"].duplicated().sum())


# --------------------------------------------------
# 3. DATE CONVERSION
# --------------------------------------------------

df["published"] = pd.to_datetime(
    df["published"],
    utc=True,
    errors="coerce"
)

print("\nDate range:")
print("Earliest:", df["published"].min())
print("Latest:", df["published"].max())


# --------------------------------------------------
# 4. RESEARCH CATEGORY ANALYSIS
# --------------------------------------------------

categories = (
    df["categories"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

category_counts = categories.value_counts()

print("\n--- TOP RESEARCH CATEGORIES ---")
print(category_counts.head(10))

plt.figure(figsize=(10, 6))

category_counts.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Research Categories")
plt.xlabel("Number of Papers")
plt.ylabel("Research Category")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "top_categories.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 5. ABSTRACT LENGTH ANALYSIS
# --------------------------------------------------

df["abstract_word_count"] = (
    df["abstract"]
    .fillna("")
    .str.split()
    .str.len()
)

print("\n--- ABSTRACT STATISTICS ---")
print(df["abstract_word_count"].describe())

plt.figure(figsize=(10, 6))

plt.hist(
    df["abstract_word_count"],
    bins=15,
    edgecolor="black"
)

plt.title("Distribution of Abstract Length")
plt.xlabel("Number of Words")
plt.ylabel("Number of Papers")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "abstract_length.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 6. PUBLICATION DATE ANALYSIS
# --------------------------------------------------

df["publication_date"] = (
    df["published"].dt.date
)

daily_papers = (
    df.groupby("publication_date")
    .size()
)

print("\n--- DAILY PUBLICATIONS ---")
print(daily_papers)

plt.figure(figsize=(10, 6))

daily_papers.plot(
    kind="bar",
    width=0.7
)

plt.title("Research Papers by Publication Date")
plt.xlabel("Publication Date")
plt.ylabel("Number of Papers")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "publication_trends.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 7. TITLE KEYWORD ANALYSIS
# --------------------------------------------------

stop_words = {
    "the", "a", "an", "and", "or", "of",
    "to", "in", "for", "with", "on", "by",
    "from", "is", "are", "using", "based",
    "via", "as", "at", "into", "their"
}

all_words = []

for title in df["title"].dropna():

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        title.lower()
    )

    words = [
        word for word in words
        if word not in stop_words
    ]

    all_words.extend(words)


word_counts = Counter(all_words)

print("\n--- TOP TITLE KEYWORDS ---")

for word, count in word_counts.most_common(15):
    print(f"{word}: {count}")


top_words = word_counts.most_common(15)

if top_words:

    words_df = pd.DataFrame(
        top_words,
        columns=["word", "frequency"]
    )

    plt.figure(figsize=(10, 6))

    words_df.sort_values("frequency").plot(
        x="word",
        y="frequency",
        kind="barh",
        legend=False,
        ax=plt.gca()
    )

    plt.title("Most Frequent Research Title Keywords")
    plt.xlabel("Frequency")
    plt.ylabel("Keyword")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "title_keywords.png",
        dpi=150
    )

    plt.close()


# --------------------------------------------------
# 8. SAVE ANALYTICS DATA
# --------------------------------------------------

processed_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "arxiv_classification_papers_eda.csv"
)

df.to_csv(processed_path, index=False)

print("\nEDA completed successfully!")
print("Charts saved in:", FIGURE_DIR)
print("Processed dataset saved in:", processed_path)
