# Intro to Hyperdimensional Computing (HDC)

Source code for a series of blog posts that explore concepts in Hyperdimensional Computing (HDC).

1. [How a Taiwanese Oolong changed the way I look at tea](https://thedataquarry.com/blog/how-a-taiwanese-oolong-changed-the-way-i-look-at-tea/)
2. [Hyperdimensional Computing (1): The algebra of hypervectors](https://thedataquarry.com/blog/hyperdimensional-computing-1/)
3. [https://thedataquarry.com/blog/hyperdimensional-computing-2/](https://thedataquarry.com/blog/hyperdimensional-computing-2/)
1. Machine learning in HDC and building a tea recommender: coming soon...

## Tea, encoded as hypervectors

This repo starts with a small, vetted dataset of 166 loose-leaf teas
from the Canadian retailer [Cha Yi](https://shop.chayi.ca/).
We're using it to demonstrate the basics of Hyperdimensional Computing (HDC) through a concrete
example, with the resulting high-dimensional data stored in
[LanceDB](https://docs.lancedb.com/).

### What we're building

HDC represents information with very wide vectors, often called hypervectors.
In this project, each one has 10,000 dimensions. Instead of asking a model to
learn a representation for us, we'll design an encoder that describes what a
tea is: its aroma, taste, class, oxidation, roast, and growing elevation. Teas
with similar properties should then land near one another in hypervector space.
The complete dataset is published in Lance format on
[Hugging Face](https://huggingface.co/datasets/prrao87/tea-hypervectors).

### The three fundamental operations

Most HDC encoders are built from three operations:

| Operation | Expression | What it does |
| --- | --- | --- |
| Binding | $\mathbf{z} = \mathbf{x} \otimes \mathbf{y}$ | Combines two hypervectors through element-wise multiplication to produce a hypervector that's different from its contributors. This lets us _associate_ one concept with another. |
| Bundling | $\mathbf{z} = \mathbf{x} \oplus \mathbf{y}$ | Superpositions two hypervectors to produce a representation that remains similar to its contributors. The operation can be repeated and each input can be weighted. |
| Permutation | $\mathbf{z} = \rho(\mathbf{x})$ | Stores ordered sequences by manipulating a hypervector's coordinates in a repeatable way, letting temporal or sequence order become part of a representation. |

The beauty of the above operations comes from the fact that they compose very well with one another, allowing us to represent and query arbitrary data in this high-dimensional space.

This project only needs binding and bundling because we're describing the
properties of a tea rather than encoding a sequence. After setup, we'll show
how those two operations combine the fields in a real record.

## Set up the environment

You'll need Python 3.13, `uv`, and a running Ollama installation.

```bash
uv sync
ollama pull nomic-embed-text
```

The public Lance dataset is the default input, so you don't need to rebuild the
source data locally. Ollama embeddings are cached after the first run.

## Turning tea fields into one hypervector

The encoder has to combine several kinds of data without flattening away the
differences between them. Binding uses element-wise multiplication to associate
a field's value vector, $`\mathbf{V}_f`$, with a separate vector for its role,
$`\mathbf{R}_f`$:

$$
\mathbf{H}_f = \mathbf{R}_f \otimes \mathbf{V}_f
$$

For aroma and taste, that gives us:
$`\mathbf{H}_{\mathrm{aroma}} = \mathbf{R}_{\mathrm{aroma}} \otimes \mathbf{V}_{\mathrm{aroma}}`$ and
$`\mathbf{H}_{\mathrm{taste}} = \mathbf{R}_{\mathrm{taste}} \otimes \mathbf{V}_{\mathrm{taste}}`$.
This is why "honey" as an aroma doesn't become interchangeable with "honey" as
a taste. Binding adds structure without adding more dimensions.

Bundling uses addition to collect the bound components into one representation
of the whole tea:

$$
\mathbf{H}_{\mathrm{tea}}
= w_{f_1}\mathbf{H}_{f_1}
\oplus w_{f_2}\mathbf{H}_{f_2}
\oplus \cdots
\oplus w_{f_n}\mathbf{H}_{f_n},
\qquad f_i \in F_{\text{present}}
$$

Here, $`F_{\text{present}}`$ contains the fields available for that record, and
$`w_f`$ is a weighting factor that controls how much a field contributes. Missing aroma or taste is left
out of the bundle, while low confidence reduces the weight given to elevation.

The sensory fields need one extra step. We lowercase, deduplicate, and sort the
aroma and taste phrases, embed each phrase with `nomic-embed-text`, then average
the 768-dimensional embeddings. If $`\mathbf{e}_f`$ is that mean and
$\mathbf{P}$ is our fixed projection matrix, we create its bipolar value
hypervector with:

$$
\mathbf{V}_f = \mathrm{sign}\left(\mathbf{e}_f\mathbf{P}\right),
\qquad f \in \{\text{aroma}, \text{taste}\}
$$

This projects up the otherwise lower-dimensional text embedding representation
to 10,000 dimensions, and the sign operation
turns every coordinate into `-1` or `+1` (i.e., makes it bipolar). The up-projection does not add new information to the text embedding. It simply translates the same semantic direction
into the higher-dimensional space where we can then bind and bundle it with the other fields.

Class gets a categorical hypervector, while oxidation, roast, and elevation use
level hypervectors that preserve order: nearby levels share more coordinates
than distant ones. Once every available property has a compatible
10,000-dimensional representation, binding supplies the structure and
bundling produces the final vector used to find similar teas.

## Build the hypervectors

Encode all 166 teas and write the resulting float16 vectors to a local LanceDB
table:

```bash
uv run src/build_hypervectors.py
```

## Check the geometry

Run controlled, one-field-at-a-time experiments:

```bash
uv run src/verify_geometry.py
```

The default encoder uses this weighted bundle:

```math
\begin{aligned}
\mathbf{H}_{\mathrm{tea}} ={}&
0.25\mathbf{H}_{\mathrm{aroma}}
\oplus 0.25\mathbf{H}_{\mathrm{taste}}
\oplus 0.16\mathbf{H}_{\mathrm{class}} \\
&{}\oplus 0.16\mathbf{H}_{\mathrm{oxidation}}
\oplus 0.03\mathbf{H}_{\mathrm{roast}}
\oplus 0.15c_{\mathrm{elevation}}\mathbf{H}_{\mathrm{elevation}}
\end{aligned}
```

These weights are domain-specific choices, not learned truths. Aroma and taste
carry half of the available weight because they describe the sensory character
of a tea. Class and oxidation intentionally overlap as strong style signals.
Elevation can contribute up to 0.15, with $`c_{\mathrm{elevation}}`$ representing our confidence in
the source value. Roast gets only 0.03 because it's a modifier and 144 of the
166 teas in this dataset have no roast. Giving it more influence would mostly
reward two teas for sharing the dataset's default value.

This checks the behavior we designed into the encoder. Nearby elevations should
be closer than distant ones, roast should be a small nudge, missing fields
should add no placeholder signal, and sensory synonyms should move less than
unrelated words.

Binding and bundling have different reversibility guarantees. This command
walks through them:

```bash
uv run src/verify_invertibility.py
```

It demonstrates exact bipolar binding, subtraction of a known bundled
component, exact recovery of a bipolar tea factor from a relationship, and the
small approximation introduced by float16 storage.

## Find similar teas

We'll use Yunnan Dian Hong, a smooth and malty black tea, as the query. Running
the default search returns its three nearest neighbours:

```bash
uv run src/search.py --tea-id 1314971975789 --limit 3
```

Expected output, abridged:

```text
Seed: Yunnan Dian Hong  (id=1314971975789)

similarity  class    title
    0.8487  black    Assam Doomni
    0.8294  black    Da Xue Shan Hong Cha
    0.8290  black    Darjeeling Namring « Tippy Muscatel » 2nd Flush

Why 'Assam Doomni' scored 0.8487:
  aroma      +0.2684
  taste      +0.2553
  class      +0.1349
  oxidation  +0.1321
  roast      +0.0054
  elevation  +0.0525
  total      +0.8487
```

Assam Doomni comes first even though it is grown in India rather than China.
The two teas share fruity, malty, caramel-like aromas and a smooth, velvety
taste. Aroma and taste contribute 0.5237 of the final score, while their shared
black class and high oxidation add another 0.2670. Roast contributes only
0.0054 because we deliberately gave it little influence. Elevation adds 0.0525:
Yunnan Dian Hong is encoded at 1,950 metres with 0.8 confidence, while Assam
Doomni is encoded at 1,600 metres with 0.6 confidence.

These contributions come directly from the weighted bundle and add back up to
the cosine score of 0.8487. They show exactly which parts of Assam Doomni's
hypervector made it the closest match.

## From similarity to associative memory

This encoder gives every tea a meaningful representation. The next step is to
connect those representations through experience.

We'll encode facts such as "this person likes this tea" as a bound relationship
vector:

$$
\mathbf{M}
= \mathbf{P}_{\text{person}}
\otimes \mathbf{R}_{\text{likes}}
\otimes \mathbf{T}_{\text{tea}}
$$

Bundling a person's interactions creates an experiential memory of the teas
they've enjoyed. Given the person and the `LIKES` relationship, we can query
that memory and recover a noisy vector pointing toward relevant teas. Comparing
it with the known tea vectors cleans up that result into recommendations.

The bundled relationships and cleanup step form an associative memory: partial
information leads us back to a stored experience. Learning here won't require
retraining a neural network. A new tasting can be added directly to the
person's memory, giving us a compact graph of people, teas, and the preferences
that connect them.
