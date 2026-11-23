
import requests
import pandas as pd

import xml.etree.ElementTree as ET
from pathlib import Path
import time

# may be delete this
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# arXiv API configuration
API_URL = "https://export.arxiv.org/api/query"

NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}


def fetch_arxiv_papers(
    search_query="cat:cs.LG",
    max_results=50
):
    """
    Fetch research paper metadata from arXiv.

    Parameters:
        search_query: arXiv search query
        max_results: Maximum number of papers

    Returns:
        List of dictionaries containing paper metadata.
    """

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    headers = {
        "User-Agent": "ResearchPaperIntelligence/1.0"
    }

    print("Connecting to arXiv API...")

    response = requests.get(
        API_URL,
        params=params,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    root = ET.fromstring(response.content)

    papers = []

    entries = root.findall("atom:entry", NAMESPACE)

    for entry in entries:

        paper_id = entry.find(
            "atom:id", NAMESPACE
        ).text.strip()

        title = entry.find(
            "atom:title", NAMESPACE
        ).text.strip()

        abstract = entry.find(
            "atom:summary", NAMESPACE
        ).text.strip()

        published = entry.find(
            "atom:published", NAMESPACE
        ).text.strip()

        authors = [
            author.find("atom:name", NAMESPACE).text
            for author in entry.findall(
                "atom:author", NAMESPACE
            )
        ]

        categories = [
            category.attrib["term"]
            for category in entry.findall(
                "atom:category", NAMESPACE
            )
        ]

        papers.append({
            "paper_id": paper_id,
            "title": " ".join(title.split()),
            "abstract": " ".join(abstract.split()),
            "authors": ", ".join(authors),
            "published": published,
            "categories": ", ".join(categories)
        })

    print(f"Collected {len(papers)} papers.")

    return papers


def save_papers(papers):

    if not papers:
        print("No papers found. Nothing to save.")
        return

    df = pd.DataFrame(papers)

    output_path = RAW_DATA_DIR / "arxiv_papers.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"Dataset saved to: {output_path}")

    print(f"Dataset shape: {df.shape}")


def main():

    papers = fetch_arxiv_papers(
        search_query="cat:cs.LG",
        max_results=50
    )

    save_papers(papers)


if __name__ == "__main__":

    main()
