
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import time


# -----------------------------------------
# PROJECT CONFIGURATION
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "arxiv_classification_dataset.csv"

API_URL = "https://export.arxiv.org/api/query"

TARGET_CATEGORIES = [
    "cs.LG",
    "cs.AI",
    "cs.CV",
    "cs.CL",
    "cs.RO"
]

TARGET_PER_CATEGORY = 300
BATCH_SIZE = 100

NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}

HEADERS = {
    "User-Agent": "ResearchPaperIntelligence/1.0"
}


# -----------------------------------------
# FETCH PAPERS FROM ARXIV
# -----------------------------------------

def fetch_category(category, target_count):

    papers = []
    seen_ids = set()

    start = 0

    print(f"\nCollecting category: {category}")

    with requests.Session() as session:

        while len(papers) < target_count:

            params = {
                "search_query": f"cat:{category}",
                "start": start,
                "max_results": BATCH_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }

            response = None

            # Retry temporary API failures
            for attempt in range(3):

                try:
                    response = session.get(
                        API_URL,
                        params=params,
                        headers=HEADERS,
                        timeout=45
                    )

                    response.raise_for_status()
                    break

                except requests.RequestException as error:

                    print(
                        f"Request failed (attempt {attempt + 1}): "
                        f"{error}"
                    )

                    if attempt == 2:
                        raise

                    time.sleep(10 * (attempt + 1))

            root = ET.fromstring(response.content)

            entries = root.findall(
                "atom:entry",
                NAMESPACE
            )

            if not entries:
                print("No more papers available.")
                break

            for entry in entries:

                paper_id_element = entry.find(
                    "atom:id",
                    NAMESPACE
                )

                primary_element = entry.find(
                    "arxiv:primary_category",
                    NAMESPACE
                )

                if paper_id_element is None or primary_element is None:
                    continue

                paper_id = paper_id_element.text.strip()

                primary_category = primary_element.attrib.get(
                    "term", ""
                )

                # Keep only the requested primary category
                if primary_category != category:
                    continue

                if paper_id in seen_ids:
                    continue

                def get_text(path):

                    element = entry.find(path, NAMESPACE)

                    if element is None or element.text is None:
                        return ""

                    return " ".join(element.text.split())

                authors = [
                    author.find(
                        "atom:name",
                        NAMESPACE
                    ).text
                    for author in entry.findall(
                        "atom:author",
                        NAMESPACE
                    )
                    if author.find(
                        "atom:name",
                        NAMESPACE
                    ) is not None
                ]

                categories = [
                    item.attrib.get("term", "")
                    for item in entry.findall(
                        "atom:category",
                        NAMESPACE
                    )
                ]

                paper = {
                    "paper_id": paper_id,
                    "title": get_text("atom:title"),
                    "abstract": get_text("atom:summary"),
                    "authors": ", ".join(authors),
                    "published": get_text("atom:published"),
                    "primary_category": primary_category,
                    "categories": ", ".join(categories)
                }

                # Exclude records without useful text
                if not paper["title"] or not paper["abstract"]:
                    continue

                papers.append(paper)
                seen_ids.add(paper_id)

                if len(papers) >= target_count:
                    break

            print(
                f"{category}: {len(papers)}/{target_count} papers"
            )

            start += len(entries)

            if len(entries) < BATCH_SIZE:
                break

            # Respect the arXiv API request interval
            if len(papers) < target_count:
                time.sleep(3)

    print(
        f"Finished {category}: {len(papers)} papers collected."
    )

    return papers


# -----------------------------------------
# BUILD THE COMPLETE DATASET
# -----------------------------------------

def main():

    all_papers = []

    for category in TARGET_CATEGORIES:

        category_papers = fetch_category(
            category,
            TARGET_PER_CATEGORY
        )

        all_papers.extend(category_papers)

    df = pd.DataFrame(all_papers)

    # Global duplicate removal
    df = df.drop_duplicates(
        subset="paper_id"
    )

    # Save dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n--- FINAL DATASET ---")

    print("Total papers:", len(df))

    print("\nPrimary category distribution:")

    print(
        df["primary_category"].value_counts()
    )

    print("\nSaved dataset to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
