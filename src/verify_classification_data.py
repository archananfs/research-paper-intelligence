
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "arxiv_classification_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)

print("\n--- PRIMARY CATEGORY DISTRIBUTION ---")
print(df["primary_category"].value_counts())

print("\n--- DUPLICATE PAPER IDS ---")
print(df["paper_id"].duplicated().sum())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- LABEL CHECK ---")
print("Unique labels:", df["primary_category"].nunique())

print("\n--- ABSTRACT LENGTH ---")
print(
    df["abstract"]
    .str.split()
    .str.len()
    .describe()
)
