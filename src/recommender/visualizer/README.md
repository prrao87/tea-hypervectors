# Tea recommender visualization data export

This package generates the JSON data consumed by the interactive Three.js visuals
in the third HDC blog post. It does not render plots or animations. It exports all
166 catalogue teas, their display metadata, and the preference learner's states
after 5, 10, and 15 tastings as one portable JSON document.

PCA is fitted once on the L2-normalized catalogue tea hypervectors. All teas and
prototype states are transformed through those same three principal components,
so the reader can move between checkpoints without the space shifting underneath
them. Recommendations are still scored by the production recommender in the
original 10,000-dimensional space; PCA controls only where items appear.

Generate the artifact from the repository root:

```bash
uv run scripts/export_recommender_visualization.py
```

The default output is
`src/recommender/visualizer/output/tea-recommender-3d.json`. Use `--copy-to` to
write the same generated JSON into a consuming site. No plotting library is
required.
