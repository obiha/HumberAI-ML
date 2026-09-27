# records.py
"""Download TV show records from the TVmaze API, aggregate them, and write a JSON summary."""
import json
import sys
from pathlib import Path

import requests

SOURCE_URL = "https://api.tvmaze.com/shows?page=0"
OUTPUT = Path("summary.json")


def fetch_records(url):
    """Download the show records from the API and return them as a list of dicts."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def shows_per_genre(records):
    """Count shows per genre. A show can list several genres, or none at all."""
    genre_counts = {}
    for show in records:
        for genre in show.get("genres") or []:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    return genre_counts


def average_rating_by_language(records):
    """Average rating per language, skipping shows that have no rating."""
    totals = {}
    counts = {}
    for show in records:
        rating = (show.get("rating") or {}).get("average")
        if rating is None:
            continue
        language = show.get("language") or "(unknown language)"
        totals[language] = totals.get(language, 0) + rating
        counts[language] = counts.get(language, 0) + 1
    return {language: round(totals[language] / counts[language], 2) for language in totals}


def shows_per_decade(records):
    """Count shows by the decade they premiered in, skipping shows with no premiere date."""
    decade_counts = {}
    for show in records:
        premiered = show.get("premiered")
        year_str = premiered.split("-")[0] if premiered else ""
        if not year_str.isdigit():
            continue
        label = f"{(int(year_str) // 10) * 10}s"
        decade_counts[label] = decade_counts.get(label, 0) + 1
    return dict(sorted(decade_counts.items()))


def build_summary(records):
    """Combine the aggregations into one dict ready to write."""
    genre_counts = shows_per_genre(records)
    return {
        "source_url": SOURCE_URL,
        "records_processed": len(records),
        "shows_per_genre": genre_counts,
        "distinct_genres": sorted(set(genre_counts)),
        "average_rating_by_language": average_rating_by_language(records),
        "shows_per_decade": shows_per_decade(records),
    }


def write_summary(summary, path):
    """Write the summary to a JSON file."""
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    """Fetch the records, build the summary, write it to disk, and report the outcome."""
    try:
        records = fetch_records(SOURCE_URL)
    except requests.exceptions.RequestException as error:
        print(f"Could not download data from {SOURCE_URL}: {error}")
        sys.exit(1)

    if not records:
        print("No records were returned from the API.")
        sys.exit(1)

    summary = build_summary(records)
    write_summary(summary, OUTPUT)
    print(f"Processed {len(records)} shows. Summary written to {OUTPUT}.")


if __name__ == "__main__":
    main()