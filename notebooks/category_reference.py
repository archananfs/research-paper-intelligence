
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.category_reference import CATEGORY_REFERENCE

df = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "arxiv_papers.csv"
)

categories = (
    df["categories"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

counts = categories.value_counts()

reference_data = []

for code, count in counts.items():

    details = CATEGORY_REFERENCE.get(
        code,
        {
            "name": "Description not yet added",
            "description": "Refer to the official arXiv taxonomy."
        }
    )

    reference_data.append({
        "Code": code,
        "Full Name": details["name"],
        "Paper Count": count,
        "Description": details["description"]
    })

reference_df = pd.DataFrame(reference_data)

reference_df = reference_df.sort_values(
    "Paper Count",
    ascending=True
)

print("\n--- RESEARCH CATEGORY REFERENCE ---")
print(
    reference_df.to_string(index=False)
)

# Create visual chart
fig, ax = plt.subplots(figsize=(13, 9))

bars = ax.barh(
    reference_df["Code"],
    reference_df["Paper Count"]
)

for bar, (_, row) in zip(
    bars,
    reference_df.iterrows()
):

    ax.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height() / 2,
        row["Full Name"],
        va="center",
        fontsize=8
    )

ax.set_title(
    "Research Categories: Codes and Full Names"
)

ax.set_xlabel("Number of Papers")
ax.set_ylabel("arXiv Category")

plt.tight_layout()

output_path = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "category_reference.png"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.show()

print("\nChart saved:", output_path)
