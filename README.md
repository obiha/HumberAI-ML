# TV Show Aggregator

This program downloads TV show records from the TVmaze API and summarizes them by genre, language and premiere decade, which is useful for spotting trends across a large catalog of shows without reading each one by hand.

## Data source

`https://api.tvmaze.com/shows?page=0` — each record is one TV show (name, genres, language, rating, network, premiere date, etc). The page returns 240 shows.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Run

    python3 records.py

## Example output

```json
"shows_per_decade": {
  "1980s": 2,
  "1990s": 8,
  "2000s": 51,
  "2010s": 179
},
"average_rating_by_language": {
  "English": 7.58,
  "Japanese": 7.88
}
```

Most of the fetched shows premiered in the 2010s, and Japanese-language shows average slightly higher ratings than English-language ones in this sample.

## Data quirks

- **Multiple or missing genres**: a show can list several genres or none. `shows_per_genre` loops over each show's genre list, so a show contributes to every genre it has, and shows with an empty list simply add to no genre count rather than being forced into a fake category.
- **Missing ratings**: some shows have no `rating.average`. Those shows are skipped in `average_rating_by_language` instead of being treated as a 0, which would have dragged averages down incorrectly.
- **Missing/malformed premiere date**: a few shows have no `premiered` date. `shows_per_decade` skips any show without a usable four-digit year rather than lumping it into a decade it didn't provide.

## Design choices

- **dict** — used for every grouping aggregation (`shows_per_genre`, `average_rating_by_language`, `shows_per_decade`) because each groups values by a key that isn't known in advance.
- **set** — used in `build_summary` to de-duplicate and sort the genre names into `distinct_genres`, since the dict keys alone don't guarantee an order to report them in.
- **list** — used for the raw records returned by the API, which are iterated in each aggregation function.

## Known limitations

- Only the first page of results (240 shows) is aggregated; paging through further pages would give a larger and more representative sample.
- Genre names are used as they appear in the API, so near-duplicate spellings (if any) are not merged.