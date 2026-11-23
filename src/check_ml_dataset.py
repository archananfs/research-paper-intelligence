
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "arxiv_papers.csv"
)

# Count all category assignments
categories = (
    df["categories"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

print("\n--- CATEGORY DISTRIBUTION ---")
print(categories.value_counts())

print("\n--- TOTAL PAPERS ---")
print(len(df))

print("\n--- UNIQUE PAPERS ---")
print(df["paper_id"].nunique())

print("\n--- ABSTRACT LENGTH ---")
print(
    df["abstract"]
    .fillna("")
    .str.split()
    .str.len()
    .describe()
)
