---
license: cc-by-nc-4.0
task_categories:
- image-classification
- text-classification
- text-retrieval
language:
- en
tags:
- lance
- lancedb
- multimodal
- tea
pretty_name: Canadian loose-leaf tea profiles in Lance format
size_categories:
- n<1K
---

# Cha Yi Tea Profiles in Lance Format

Dataset of loose-leaf teas for multimodal research and experiments in LanceDB.
The data is stored in Lance format, including normalized JPEG bytes and vetted
tea metadata.

## Dataset description

This dataset contains 166 English-language loose-leaf tea profiles collected
from the public storefront of [Maison de thé Cha Yi](https://chayi.ca/), a tea
vendor based in Gatineau, Quebec, Canada. It covers the store's Oolong, green,
white, black, and yellow tea collections.

The data is stored in Lance format, and searched using [LanceDB](https://docs.lancedb.com/),
a multimodal lakehouse for AI.

Each row combines product text, a tea-class label, sensory and processing
labels, region and elevation metadata, a source URL, and the primary product
image stored as bytes. The dataset is distributed as a ready-to-query local
Lance table with a full-text search index over the product description.

This is a small, point-in-time research snapshot. It is not an official Cha Yi
dataset or a live copy of the store catalog.

### Dataset summary

| Tea class | Rows |
| --- | ---: |
| Black | 51 |
| Green | 65 |
| Oolong | 35 |
| White | 12 |
| Yellow | 3 |
| **Total** | **166** |

- **Language:** English
- **Modalities:** Text, images, tabular metadata
- **Format:** Lance
- **Dataset path:** `data/train.lance`
- **Rows:** 166
- **Hub split:** `train` (the complete 166-row snapshot)
- **Geographic context:** Canadian retailer; teas originate from multiple
  tea-producing regions
- **Metadata collected:** July 26, 2026
- **Vetted snapshot finalized:** July 28, 2026

## Dataset structure

The Hugging Face repository contains the Lance dataset at
`data/train.lance/`, alongside this dataset card at root-level `README.md`.
This matches the structure used by
[`lance-format/textvqa-lance`](https://huggingface.co/datasets/lance-format/textvqa-lance/tree/main).

### Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | `int64` | Stable product identifier from the source storefront |
| `source_url` | `string` | Direct link to the corresponding Cha Yi product page |
| `image` | `binary` | Bytes of the normalized JPEG product image |
| `class` | `string` | One of `black`, `green`, `oolong`, `white`, or `yellow` |
| `title` | `string` | English product display title |
| `description` | `string` | English product description and tasting or brewing text |
| `country` | `string`, nullable | Tea origin country when available |
| `region` | `string` | Vetted growing-region label |
| `oxidation` | `string` | Derived oxidation label |
| `roast` | `string` | Derived roast label |
| `aroma` | `list<string>` | Description-grounded aroma spans |
| `taste` | `list<string>` | Description-grounded taste spans |
| `elevation_meters` | `int64` | Vetted representative growing elevation |
| `elevation_confidence` | `float64` | Confidence in the elevation value, from 0 to 1 |

The included named full-text search index is `description_fts`, built on
`description`.

### Data splits

The complete snapshot is exposed as the Hub's `train` split so the dataset
viewer can render it. This is a serving convention, not a predefined evaluation
split. Users who evaluate models should define and report their own split
strategy. Be mindful that related teas, regional styles, and naming patterns can
make a random row-level split easier than a real-world generalization task.

## Load and query the dataset

Open the LanceDB table directly from the Hub:

```python
import lancedb

db = lancedb.connect(
    "hf://datasets/prrao87/tea-hypervectors/data"
)
table = db.open_table("train")

rows = (
    table.search()
    .select(
        [
            "title",
            "class",
            "description",
            "region",
            "elevation_meters",
            "source_url",
        ]
    )
    .limit(5)
    .to_list()
)
print(rows)
```

Filter and project metadata without materializing the binary image column:

```python
oolongs = (
    table.search()
    .where("class = 'oolong'")
    .select(
        [
            "id",
            "title",
            "class",
            "description",
            "region",
            "elevation_meters",
            "source_url",
        ]
    )
    .limit(3)
    .to_list()
)
print(oolongs)
```

The repository uses LanceDB rather than a standard Parquet-backed Hugging Face
`datasets` configuration, so it may not have the usual Dataset Viewer preview.

## Dataset creation

### Source data

All records were derived from public pages and public collection data on the
Cha Yi storefront. Every row retains a `source_url` for traceability and for
checking the current product page.

The extraction process:

1. Discovered products in five public tea collections.
2. Requested each public product page with English Canadian language
   preferences.
3. Kept English display text and class information.
4. Derived oxidation, roast, aroma, and taste labels.
5. Estimated and manually vetted region/elevation metadata.
6. Downloaded and normalized one JPEG image per exported tea.
7. Used Polars to select the serving schema and attach local image bytes.
8. Ingested the Polars DataFrame into LanceDB.
9. Built a named full-text search index over `description`.

The metadata extraction made 171 sequential requests over approximately 172
seconds. All collection, product-page, retry, and image requests were separated
by at least one second. Cached normalized JPEGs were reused during repeat
ingestions.

### Language processing

The extractor requested `locale=en` and sent `Accept-Language: en-CA`. It did
not translate source content. Text detected as French was omitted from the
language fields, while source URLs and public handles were preserved even when
they contained French words.

The source storefront's tag list is partly French. The richer source export
therefore retained only a conservative allowlist of unambiguously English tags
and proper tea or place names. The LanceDB table contains the smaller serving
schema documented above rather than the complete extraction schema.

### Labels

The `class` label comes from the one selected storefront collection associated
with each product:

- `black`
- `green`
- `oolong`
- `white`
- `yellow`

These labels are useful for experiments but should not be treated as a
comprehensive or authoritative tea taxonomy.

### Personal and sensitive information

The dataset consists of public commercial product information. It was not
designed to include personal, private, or sensitive information.

## Intended uses

This dataset is intended for non-commercial educational and research work,
including:

- multimodal and text retrieval experiments;
- tea-type image or text classification;
- feature engineering and metadata enrichment;
- full-text, vector, or hybrid search prototypes;
- demonstrations of storing binary images alongside structured metadata in
  LanceDB.

Users should preserve source attribution and follow each row's `source_url`
when current product information matters.

## Limitations

- The dataset represents one retailer and a small catalog, so it is not
  representative of the global tea market.
- The five classes are imbalanced, with only three yellow teas.
- Labels follow storefront collection membership rather than an independently
  validated botanical or production taxonomy.
- Descriptions and photography were written and selected for retail use and may
  encode the vendor's editorial and marketing perspective.
- Prices, availability, descriptions, brewing guidance, and images may have
  changed since the snapshot was collected.
- English-only filtering removes some of the source storefront's multilingual
  information.
- Product images can include differences in composition, lighting, packaging,
  and presentation that models may learn instead of tea-specific features.
- There are no predefined splits or benchmark metrics.

## Licensing and attribution

The dataset curation is released under the
[Creative Commons Attribution-NonCommercial 4.0 International
license](https://creativecommons.org/licenses/by-nc/4.0/) to the extent that it
is licensable by the dataset maintainer.

All underlying intellectual property—including tea names, tasting
descriptions, vendor branding, and product photography—remains the property of
**Maison de thé Cha Yi** and its respective rights holders. The dataset license
does not relicense or override those underlying rights.

- **Original source:** [Maison de thé Cha Yi](https://chayi.ca/)
- **Record-level attribution:** Each row contains a direct `source_url`
- **Permitted purpose:** Non-commercial education and research

If you use the dataset, retain attribution, link back to the original product
pages, and consider supporting the vendor by buying their products online, and
spreading the word on social media.

## Citation

There is no formal academic citation for this dataset. If you use it, please cite the dataset repository and acknowledge Maison de thé Cha Yi as the copyright holder of the
underlying product content. For example:

```bibtex
@dataset{chayi_tea_lancedb_2026,
  title        = {Cha Yi Tea Profiles in LanceDB},
  year         = {2026},
  month        = {7},
  publisher    = {Hugging Face},
  note         = {Underlying product content and photography from Maison de thé Cha Yi},
  url          = {https://huggingface.co/datasets/prrao87/tea-hypervectors}
}
```
