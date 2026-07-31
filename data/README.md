---
language:
  - en
license: cc-by-nc-4.0
pretty_name: Canadian Loose-Leaf Tea Profiles
size_categories:
  - n<1K
task_categories:
  - image-classification
  - text-classification
  - text-retrieval
tags:
  - tea
  - canada
  - image
  - text
  - tabular
  - multimodal
  - lancedb
---

# Canadian Loose-Leaf Tea Profiles

This directory contains a small, reproducible research dataset built from the
public storefront of [Cha Yi](https://chayi.ca/), a Canadian tea vendor based in
Gatineau, Quebec.

## Contents

- `extract.py` — the polite extractor.
- `enrich.py` — derives oxidation, roast, aroma, and taste with
  focused DSPy signatures.
- `estimate_elevation.py` — extracts growing regions and estimates their
  elevations with DSPy native tool calling and Tavily.
- `download_images.py` — respectfully downloads, resumes, and normalizes every
  primary image to `img/<id>.jpg`.
- `normalize_images.py` — standalone JPEG normalization/validation utility used
  by the downloader.
- `ingest.py` — reads the vetted final JSON with Polars and builds local
  LanceDB table `tea-db`.
- `index.py` — creates the table's named full-text search index.
- `query.py` — runs bounded FTS queries without reading image bytes.
- `extracted/tea_data.json` — the full flat export of Oolong, green, white,
  black, and yellow tea collections.
- `extracted/summary.json` — extraction timing, counts, and null summary.
- `extracted/tea_data_enriched.json` — products with the derived ontology
  fields appended.
- `extracted/tea_data_final.json` — enriched products with region, estimated
  elevation in metres, and elevation confidence appended.
- `../img/` — the 166 resumable, normalized JPEG source images.
- `tea-db/` — the local LanceDB database containing the `tea-db` table.

## What the extractor does

It uses the five public collection endpoints only to discover product handles,
then follows each localized product page. It persists a small, flat record built
from the rendered English title and description plus the page's embedded
localized product JSON:

| Field | Description |
| --- | --- |
| `id` | Shopify product identifier |
| `source_url` | Canonical Cha Yi product URL |
| `category` | Collection key: `oolong`, `green`, `white`, `black`, or `yellow` |
| `title` | Rendered localized product title |
| `description` | Rendered localized product description and brewing text |
| `country` | Canonical country when explicit in localized tags or vendor text |
| `image_url` | Primary image URL from the embedded product JSON |

The extractor sends `Accept-Language: en-CA` and requests `locale=en`. It does
not reject accented characters or French source values. It also does not infer
oxidation, roast, or sensory ontology; those belong in a later evidence-backed
enrichment stage.

## Enrich the tea ontology

The enrichment script uses OpenRouter's `google/gemini-3.5-flash` through DSPy.
Set `OPENROUTER_API_KEY` in `.env`, inspect the exact signature instructions,
then run the six-tea validation sample:

```bash
uv run data/enrich.py --show-signatures
uv run data/enrich.py \
  --limit-per-category 3 \
  --max-concurrency 2 \
  --no-cache
```

For each tea, the compound module first calls the description-grounded
predictors concurrently with DSPy's async API. When processing roast is absent
from the description, a second signature infers it from the title, category,
and tea knowledge. The flat `roast_basis` field records whether the result came
from `description` or `tea_knowledge`. Aroma and taste values must be exact
substrings of the source description; the script rejects paraphrased or
invented spans. The sample command writes
`extracted/tea_data_enriched.json`, preserving the original product fields and
appending the derived fields directly to each product. It does not modify the
scraped metadata.

After the validation sample passes, enrich the complete export:

```bash
uv run data/enrich.py --all
```

## Estimate growing-region elevation

Run this stage only after `extracted/tea_data_enriched.json` exists. Set
`TAVILY_API_KEY` in `.env`, then inspect the three DSPy signatures without
calling the model or Tavily:

```bash
uv run data/estimate_elevation.py --show-signatures
```

The program uses `dspy.Predict` with `dspy.Tool` and `dspy.ToolCalls`; it does
not use ReAct. It first extracts a region for each tea, deduplicates normalized
`(region, country)` pairs, asks the model for exactly one native tool call per
unique location, executes a one-credit Tavily `basic` search, and estimates a
representative elevation from the search snippets. Elevations must be integer
multiples of 50 metres; other integer estimates are rounded down to the nearest
multiple of 50 during validation. Confidence is constrained to 0–1.
When a product description explicitly states its elevation, that
product-specific value takes precedence and the program skips both tool
planning and Tavily for that product. Approximate values such as “almost 2700
meters” retain the stated number; ranges use their midpoint before rounding.
Final product records retain debugging provenance: `elevation_basis`,
`elevation_raw_meters`, matched `elevation_description_evidence`, the proposed
`elevation_tool_call`, the exact `elevation_search_query`, and Tavily
`elevation_search_results` containing titles, URLs, snippets, and scores.
Search fields are null or empty for description-backed elevations.
The default model for this stage is the inexpensive, tool-capable
`google/gemini-2.5-flash-lite` through OpenRouter.

Tavily results are persisted in
`extracted/elevation_search_cache.json`. This explicit cache is separate from
DSPy's LM cache and prevents repeated locations from spending another Tavily
credit across products or reruns.

Test exact products before processing all 166 entries:

```bash
uv run data/estimate_elevation.py \
  --titles "Bai Ya Qi Lan" "Assam Dejoo" "Da Yu Ling" "Takachiho Koshoun" \
  --output /tmp/tea_data_final_sample.json
```

After reviewing that sample, process the complete enriched export:

```bash
uv run data/estimate_elevation.py --all
```

To inspect named products instead of the default green/black sample:

```bash
uv run data/enrich.py \
  --titles "Muzha Tie Guan Yin" "Ali Shan - Roasted" \
  --output /tmp/tea_data_enriched.json
```

Every pull starts at least one second after the preceding one. This applies to
collection discovery, product pages, and retries. The full run makes five
collection-data pulls plus one public product-page pull per tea: 171 requests
over about three minutes at the current catalog size.

## Run it

From the repository root:

```bash
uv run data/extract.py \
  --collections black green \
  --limit-per-collection 3 \
  --output /tmp/tea_data.json
uv run data/extract.py --limit 0
uv run data/download_images.py
uv run data/ingest.py
uv run data/index.py
uv run data/query.py
uv run ruff format \
  data/extract.py data/enrich.py data/estimate_elevation.py \
  data/download_images.py data/normalize_images.py data/ingest.py
```

The first command is the six-product validation gate. The second performs the
full metadata export. In the July 26, 2026 run, all 166 records had IDs, source
URLs, categories, titles, descriptions, and image URLs. Country was explicit
for 164 records; the two nulls were multi-origin blends.

## Polars and LanceDB serving schema

`download_images.py` reads IDs and image URLs with Polars, waits at least one
second between remote CDN requests, checkpoints after every product, and
normalizes each result to a validated `img/<id>.jpg`. Existing JPEGs are reused
without network traffic. Genuine HEIC/HEIF payloads are decoded and converted;
JPEG payloads served under `.heic` URLs are renamed without recompression.

`ingest.py` reads `extracted/tea_data_final.json` with Polars, projects the
agreed fields, attaches bytes from the normalized local JPEGs, and passes the
Polars DataFrame directly to LanceDB. The table columns are:

`id`, `source_url`, `image`, `class`, `title`, `description`, `country`,
`region`, `oxidation`, `roast`, `aroma`, `taste`, `elevation_meters`, and
`elevation_confidence`.

The local table contains 166 rows and is optimized after ingestion. Run
`uv run data/ingest.py --overwrite` to intentionally replace an existing local
table.

## Create and query the FTS index

Run `index.py` only after the LanceDB table exists, and run `query.py` only
after the FTS index exists:

```bash
uv run data/index.py
uv run data/query.py
```

The default query is `roasted oolong high mountain`. `query.py` passes query
text directly to LanceDB FTS without expansion or other rewriting. Pass another
query as the positional argument:

```bash
uv run data/query.py "charcoal roasted"
```

For the top three high-mountain oolongs, combine literal FTS with an explicit
class filter:

```bash
uv run data/query.py "high mountain oolong" --class oolong --limit 3
```

| Rank | Tea | Class | FTS score |
| ---: | --- | --- | ---: |
| 1 | [Shan Lin Shi](https://shop.chayi.ca/products/shan-lin-shi) | oolong | 7.6560 |
| 2 | [Mi Lan Xiang 2024 (-20%)](https://shop.chayi.ca/products/mi-lan-xiang-2024-20-de-rabais) | oolong | 6.5508 |
| 3 | [Hong Xiang](https://shop.chayi.ca/products/hong-xiang) | oolong | 5.5710 |

The current named index, `description_fts`, covers `description`. LanceDB's
native FTS indexes cover one text column each, so future `title`, scalar, or
vector indexes can be added independently in `index.py`. Use
`uv run data/index.py --replace` to intentionally rebuild the current index.

If the database is missing, both scripts print the complete prerequisite
sequence: extract the metadata, ingest it into LanceDB, and then create the FTS
index.

## Dataset summary

The dataset contains 166 English-language tea profiles from Cha Yi's public
Oolong, green, white, black, and yellow tea collections. Each profile includes
source-linked product metadata, an English title and description, sensory and
processing labels, vetted region/elevation fields, and a primary product image.

| Class | Products |
| --- | ---: |
| Black | 51 |
| Green | 65 |
| Oolong | 35 |
| White | 12 |
| Yellow | 3 |
| **Total** | **166** |

## Intended uses and limitations

This dataset is intended for non-commercial educational and research work,
including multimodal retrieval, classification, feature engineering, metadata
enrichment, and vector-search experiments. It is a point-in-time export rather
than a live catalog: prices, availability, descriptions, and images may change
at the source. Always follow each record's `source_url` for the current product
page.

The dataset has no official train, validation, or test split. Its class labels
come from the five selected storefront collections and should not be treated as
a comprehensive tea taxonomy. The text is English-only, but product handles and
source URLs may retain French words used by the original storefront.

## Upload to Hugging Face

The following sequence irreversibly deletes the existing Hub repository,
recreates it as a public dataset, and uploads the current LanceDB snapshot and
dataset card:

```bash
hf auth login
hf auth whoami
hf repos delete prrao87/tea-hypervectors --repo-type dataset
hf repos create prrao87/tea-hypervectors --repo-type dataset

hf upload prrao87/tea-hypervectors \
  data/tea-db/tea-db.lance \
  data/train.lance \
  --repo-type dataset \
  --commit-message "Upload vetted Lance dataset"

hf upload prrao87/tea-hypervectors \
  data/HF_DATASET_CARD.md \
  README.md \
  --repo-type dataset \
  --commit-message "Add updated dataset card"
```

The uploads place only the Lance dataset and dataset card on the Hub. The
repository contains `data/train.lance/` and root-level `README.md`, matching
[`lance-format/textvqa-lance`](https://huggingface.co/datasets/lance-format/textvqa-lance/tree/main).
The local source for `data/train.lance/` is the complete
`data/tea-db/tea-db.lance/` directory. Its `data/`, `_versions/`,
`_transactions/`, `_indices/`, and manifest files are preserved recursively.
The separate `data/tea-db/__manifest/` directory belongs to the parent LanceDB
namespace and is not uploaded. Repository code, raw JSON, and `img/` are never
upload sources.

## Intellectual property and source attribution

This dataset is published strictly for **non-commercial, educational, and AI
research**, such as multimodal retrieval, classification, feature engineering,
and vector-search experiments.

> [!NOTE]
> All underlying intellectual property—including tea names, tasting
> descriptions, vendor branding, and product photography—remains the sole
> copyright of **Maison de thé Cha Yi**. This repository makes no claim of
> ownership over the underlying text, tasting notes, or product photography.

- **Original source:** [https://chayi.ca/](https://chayi.ca/)
- **Original product links:** Each record includes a `source_url` pointing
  directly to Cha Yi's official store page.
- **Support the vendor:** If you use or enjoy this dataset, please support Cha
  Yi by purchasing their tea or sharing their store with others.

## Licensing

The original code and dataset curation in this repository are published under
the **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)** license to the extent they are licensable by the dataset
maintainer. This license does **not** relicense or override Cha Yi's rights in
the source descriptions, tasting notes, names, branding, or product images.
