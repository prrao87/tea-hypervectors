# What we learned building a tea encoder with HDC

This is a durable record of the main ideas, design decisions, experiments, and
caveats that emerged while building the tea hypervector project. It is meant to
preserve the reasoning behind the code, not merely describe the files.

## The project in one paragraph

We started with a vetted dataset of 166 loose-leaf teas and designed a typed
Hyperdimensional Computing (HDC) encoder for similarity search. Each tea becomes
a 10,000-dimensional MAP hypervector built from aroma, taste, class, oxidation,
roast, and growing elevation. Nomic supplies the semantic signal for the two
text fields; HDC supplies the structure needed to combine text, categories,
ordinals, and numbers in one inspectable representation. The resulting vectors
are stored in LanceDB and now also feed a small online preference learner.

The central lesson was not that HDC makes text embeddings obsolete. It was that
flattening structured data into prose delegates its structure to the embedding
model, while a typed HDC encoder makes our structure and domain assumptions
explicit, testable, and directly manipulable.

## The HDC mental model

The project uses two of HDC's three familiar operations:

- **Binding**, implemented in MAP as element-wise multiplication, associates a
  value with a role. For example, binding `honey` to the `aroma` role makes it
  different from `honey` bound to the `taste` role.
- **Bundling**, implemented here as element-wise addition, collects several
  component hypervectors into a representation that remains similar to them.
  Components can cast differently weighted votes.
- **Permutation** represents order or sequences. We did not need it to describe
  a tea record, although it may become useful for temporal tasting histories.

For a field (f), the typed component is:

$$
\mathbf{X}_f = \mathbf{R}_f \odot \mathbf{V}_f
$$

The complete tea is a weighted bundle of the components that are actually
present:

```math
\mathbf{H}
= 0.25\mathbf{A}
\oplus 0.25\mathbf{T}
\oplus 0.16\mathbf{C}
\oplus 0.16\mathbf{O}
\oplus 0.03\mathbf{R}
\oplus 0.15c_E\mathbf{E}
```

These weights are a domain hypothesis, not learned truth:

- Aroma and taste jointly carry half the signal because they describe sensory
  character. They remain separate roles.
- Class and oxidation each receive 0.16. Their overlap is intentional: class is
  meaningful, while oxidation distinguishes teas within a class, especially
  oolongs.
- Elevation contributes at most 0.15. Its actual weight is multiplied by the
  confidence (c_E), so uncertain evidence speaks more quietly.
- Roast receives only 0.03. It modifies a tea rather than defining it, and 144
  of the 166 records are unroasted. A larger weight would mostly reward a shared
  default.

Encoder design is therefore an act of modeling. Domain knowledge and explicit
heuristics matter at least as much as the vector operations.

## Giving each data type the right geometry

The encoder does not pretend that every field is text:

| Field | Representation | Intended geometry |
| --- | --- | --- |
| Aroma | Nomic embedding, projected and bound to `aroma` | Semantic |
| Taste | Nomic embedding, projected and bound to `taste` | Semantic |
| Class | Random categorical item vector | Nominal; no ordering |
| Oxidation | TorchHD level vector | Ordered: low, medium, high |
| Roast | TorchHD level vector | Ordered: none, light, medium, heavy |
| Elevation | 61 level vectors from 0–3,000 m | Ordered at 50 m resolution |

Aroma and taste phrases are lowercased, normalized, deduplicated, and sorted.
Each phrase is embedded with `nomic-embed-text`; phrase vectors are averaged and
L2-normalized. A fixed Rademacher projection maps the 768-dimensional semantic
direction into 10,000 dimensions:

$$
\mathbf{V}_f = \mathrm{sign}(\mathbf{e}_f\mathbf{P})
$$

The projection adds no information. It translates the Nomic direction into a
bipolar MAP representation that can participate in binding and bundling.

Missing sensory fields are omitted rather than replaced with a shared
`missing` vector. Their weights are not redistributed, because doing so would
turn absent evidence into an amplification of whatever field survived. An
elevation of zero is a real value; only `None` means missing. Elevation
confidence changes the component's weight, not its direction.

## Determinism without vocabulary bookkeeping

Atomic hypervectors are generated from stable hashes of:

```text
global seed | namespace | token
```

This means adding a new symbol does not disturb any existing symbol, and there
is no vocabulary file to synchronize. The name `HypervectorFactory`, exposed as
`encoder.hv`, was deliberately chosen over `SymbolMemory`: it describes what
the object does without introducing associative-memory terminology too early.

The manifest is intentionally small and readable. Earlier ideas involving
fingerprints, update histories, hand-maintained vote bounds, and multiple opaque
weight profiles were removed. This is a teaching project and a single snapshot
of the encoded data; it does not need infrastructure for continual model
versioning.

## MAP reversibility: the nuance that mattered most

“Binding and bundling are reversible” is too imprecise for MAP. There are three
different claims:

### 1. Bipolar binding is exactly self-inverse

For coordinates in \(\{-1,+1\}\), a key is its own inverse:

$$
(\mathbf{x}\odot\mathbf{k})\odot\mathbf{k}=\mathbf{x}
$$

because every coordinate of \(\mathbf{k}\odot\mathbf{k}\) is 1. This does not
hold for an arbitrary weighted vector.

### 2. A known bundled component can be subtracted

The tea bundle is deliberately kept as an unnormalized float32 sum. If a
component and its weight are known, it can be removed and replaced:

$$
\mathbf{H}'
= \mathbf{H}
{}- w_E\mathbf{E}(1400)
{}+ w_E\mathbf{E}(1600)
$$

This is exact to ordinary float32 rounding. It does **not** mean an unknown
component can be discovered from the bundle alone. Addition superposes signals;
it does not preserve a list of its operands.

Normalizing the bundle to bipolar signs would destroy the weight magnitudes and
the ability to subtract a known weighted component. The search vector therefore
remains an unnormalized weighted sum; cosine similarity handles its magnitude.

### 3. Relationship factors need a bipolar form

The weighted tea bundle is excellent for similarity but is not self-invertible
under binding. For an exact subject–predicate–object demonstration, the encoder
creates a separate bipolar tea factor:

$$
\mathbf{T}_{\pm}=\mathrm{sign}(\mathbf{H})
$$

Then:

$$
\mathbf{M}
= \mathbf{P}_{person}
\odot \mathbf{R}_{likes}
\odot \mathbf{T}_{\pm}
$$

can be unbound exactly when the other bipolar factors are known. What is
recovered is \(\mathbf{T}_{\pm}\), not the original weighted tea bundle;
bipolarization discarded those magnitudes. Bundling many relationships creates
cross-talk, so recovery from an associative memory is approximate and needs a
cleanup search against known items.

That separation—weighted bundle for similarity, bipolar factor for exact
binding—is the cleanest way we found to teach MAP honestly.

## Storage boundary: float16 at rest, float32 for HDC math

The 10,000-dimensional bundles are persisted in LanceDB as float16 arrays,
cutting each vector payload from 40 KB to 20 KB. Every vector is widened to
float32 before TorchHD performs binding, bundling, subtraction, normalization,
or similarity calculations.

Float16 persistence is approximate, not bit-exact. The verification suite found
that all reloaded vectors retained essentially the same direction and that the
nearest neighbour was unchanged for all 166 teas. Storage precision should be
judged by whether the relevant geometry and rankings survive, not whether every
bit round-trips.

## The clearest similarity example

Yunnan Dian Hong retrieves Assam Doomni as its nearest neighbour with cosine
similarity 0.8484. The score decomposes into:

| Component | Contribution |
| --- | ---: |
| Aroma | +0.2685 |
| Taste | +0.2551 |
| Class | +0.1348 |
| Oxidation | +0.1322 |
| Roast | +0.0054 |
| Elevation | +0.0524 |
| **Total** | **+0.8484** |

This was a lovely result because the teas come from different countries, yet
share fruity, malty, caramel-like aromas and smooth, velvety tastes. The
component contributions add back to the cosine score, making the retrieval
inspectable rather than merely plausible.

## The counterfactual elevation experiment

Ali Shan is encoded at 1,400 m. We asked for a tea like Ali Shan but 200 m
higher by replacing only its elevation component:

$$
\mathbf{H}_{query}
= \mathbf{H}_{Ali\ Shan}
{}- 0.15\mathbf{E}(1400)
{}+ 0.15\mathbf{E}(1600)
$$

Mi Lan Xiang Lao Cong, grown at 1,600 m, moved from second place to first. A
1,000 m Dong Ding fell from first to third. This demonstrated a useful property
of the typed representation: one field can be changed while the other field
components remain untouched.

This arithmetic was a focused experiment, not part of the current natural-
language retrieval interface. The present search command accepts a tea ID and
uses its original bundle. A future query layer would need to parse natural
language into structured intent such as:

```json
{
  "seed_tea": "Ali Shan",
  "field": "elevation_meters",
  "operation": "offset",
  "value": 200
}
```

It could then reconstruct the deterministic old and new elevation components
and send the modified vector to the existing LanceDB search. If the request
requires an exact elevation constraint rather than counterfactual similarity,
that should be a metadata filter, not a similarity signal.

## What the text-embedding comparison actually showed

We compared HDC with a fair flat Nomic baseline containing the same fields:
aroma, taste, class, oxidation, roast, elevation, and elevation confidence. We
did not give either representation extra title, description, country, or region
information.

Changing Ali Shan's full text from `1400 metres` to `1600 metres` barely changed
the ranking. We then tried text-vector arithmetic:

$$
\mathbf{q}
= \mathrm{norm}\left(
\mathbf{e}_{record}
{}- \mathbf{e}_{1400}
{}+ \mathbf{e}_{1600}
\right)
$$

The honest result was mixed. A bare-number formulation happened to move the
1,600 m tea to first place. Equally plausible typed-field and natural-language
formulations did not at their natural scale; one required an extra hand-tuned
multiplier. Across all 166 teas, asking for 200 m higher produced:

| Method | Mean movement of top-5 elevations | Change in target MAE@5 |
| --- | ---: | ---: |
| Typed HDC component replacement | **+56.0 m** | **−13.5 m** |
| Nomic direct string replacement | −1.1 m | +1.2 m |
| Nomic per-query bare-number arithmetic | −17.3 m | +10.8 m |
| Nomic per-query typed-field arithmetic | −5.1 m | +12.5 m |
| Nomic per-query natural-language arithmetic | −2.6 m | −2.1 m |

Lower target MAE is better. Nomic arithmetic often changed rankings, but the
movement was not reliably in the requested direction. The reason is that a
pooled text embedding is not guaranteed to decompose additively:

$$
\mathbf{e}(\text{complete record})
\ne
\mathbf{e}(\text{other fields})
{}+ \mathbf{e}(\text{elevation})
$$

This experiment does not prove that text embeddings cannot do vector
arithmetic, nor that HDC universally retrieves better neighbours. A developer
could build a query rewriter, metadata filters, a reranker, or another typed
scoring system that reproduces the behavior. The defensible conclusion is:

> Text arithmetic may work for a particular phrasing, while typed HDC makes the
> intervention explicit, reproducible, and directionally meaningful by design.

If semantic search plus metadata filtering already solves an application, HDC
may be unnecessary complexity. Here, HDC is valuable because the goal is to
teach representation design, inspect individual signals, manipulate them, and
eventually build memory with the same algebra.

## Geometry checks are tests, not the story

The synthetic checks verify that the implementation obeys the manifest:

- nearby elevations are closer than distant elevations;
- adjacent oxidation levels are closer than distant ones;
- roast is a smaller nudge than class or oxidation;
- missing fields add no placeholder signal;
- field and phrase ordering does not change the result;
- `honey` bound as aroma differs from `honey` bound as taste.

These are important correctness tests, but several are true because we designed
the weights or geometry that way. They should not be presented as proof of
superiority over learned text representations. The better narrative uses two
end-to-end stories: Yunnan Dian Hong → Assam Doomni for explainable similarity,
and Ali Shan +200 m for controllable counterfactual retrieval. A compact
aggregate result protects the latter from looking cherry-picked without turning
the post into an ablation study.

## The preference learner and experiential memory

The project now contains a working online preference learner. It reuses the
stored tea bundles without modifying or retraining the encoder.

Each catalog vector is L2-normalized so records with more populated fields do
not cast louder votes merely because their raw bundles have larger norms. Liked
teas are accumulated into a positive prototype, disliked teas into a negative
prototype, and neutral teas are remembered only as already tried:

$$
\mathbf{S}^{+}_{new}
= \mathbf{S}^{+}_{old}
{}+ w_{new}\mathbf{x}_{new}
$$

The finalized prototypes are normalized. An untried tea receives:

$$
\mathrm{score}(j)
= \mathbf{x}_j\cdot\mathbf{p}^{+}
{}- \lambda(\mathbf{x}_j\cdot\mathbf{p}^{-})
$$

The default negative penalty is \(\lambda=0.25\), deliberately cautious while
the history contains only one disliked tea. Learning is a weighted vector
addition—no backpropagation or encoder retraining is needed. The append-only CSV
remains the source of truth, and replaying it makes the result reproducible.

At the latest verified state, the history contained 14 tastings: seven liked,
one disliked, and six neutral. The positive prototype had 12 votes and the
negative prototype had two. The leading untried recommendations were:

1. Shan Lin Shi — 0.7097
2. Dong Ding - Charcoal roasted — 0.7088
3. Mi Lan Xiang Lao Cong — 0.7058
4. Mi Lan Xiang — 0.7013
5. Ya Shi Xiang — 0.6976

All six recommender tests passed. This prototype learner is the currently
implemented form of experiential memory. The subject–predicate–object binding
example remains the next, more explicitly associative-memory direction; it
should not be described as already powering the recommender.

## The broader lessons

1. **Choose geometry by data type.** Nominal categories, ordered variables,
   numbers, and text semantics should not all be encoded the same way.
2. **Binding adds roles; bundling collects evidence.** Confusing those jobs
   makes both the implementation and the explanation harder.
3. **Weights encode domain judgment.** They should be visible, justified, and
   treated as hypotheses.
4. **Missingness is absence, not automatically a feature.** A shared missing
   token can manufacture similarity between incomplete records.
5. **Do not normalize away information you need later.** The weighted tea sum
   stays unnormalized so known components can be removed or replaced.
6. **Be exact about reversibility.** Bipolar binding is exactly self-inverse;
   known addends can be subtracted; unknown bundled contents require cleanup;
   bipolarizing a weighted bundle is lossy.
7. **Separate compute precision from storage precision.** Float16 is useful at
   rest, while float32 keeps TorchHD arithmetic safe and understandable.
8. **Prefer determinism over bookkeeping.** Stable hash-derived symbols avoid a
   vocabulary synchronization problem.
9. **A surprising example is not a universal benchmark.** Use it to explain a
   capability, then add one compact aggregate check and state the limits.
10. **HDC is not magic.** Its value here is explicit composition,
    inspectability, counterfactual control, and sample-efficient online memory.

## Where the important pieces live

- `src/hdc/manifest.py` — dimensions, vocabularies, levels, and weights.
- `src/hdc/core.py` — deterministic atomic vectors and MAP primitives.
- `src/hdc/projection.py` — Nomic-to-bipolar random projection.
- `src/hdc/encoder.py` — typed components and the complete tea bundle.
- `src/hdc/storage.py` — float16 persistence and float32 restoration.
- `src/verify_geometry.py` — controlled correctness checks.
- `src/verify_invertibility.py` — the exact reversibility boundaries.
- `src/search.py` — LanceDB similarity search and contribution breakdown.
- `src/recommender/` — preference events, prototypes, and ranking.
- `inputs/preferences/tastings.csv` — append-only experiential history.
- `data/tea-hv-db/` — the local LanceDB table.

The public source dataset is available in Lance format at
<https://huggingface.co/datasets/prrao87/tea-hypervectors>.

