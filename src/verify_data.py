
import pandas as pd
from pathlib import Path
# may be delete this

file_path = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "arxiv_papers.csv"
)

df = pd.read_csv(file_path)

print("\n--- DATASET INFORMATION ---")
print(df.info())

print("\n--- FIRST 5 PAPERS ---")
print(df[["title", "published", "categories"]].head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATES ---")
print(df.duplicated(subset=["paper_id"]).sum())

print("\n--- DATASET SHAPE ---")
print(df.shape)
