# Instructions

- Run all Python scripts through `uv run`; do not invoke `python` or `python3`
  directly.
- Use `pytest` for all tests; do not use `unittest`.
- Format changed Python files with `uv run ruff format <files>` and run affected pytest cases when behavior changes. Documentation-only edits need content checks; broaden testing only for a concrete coverage gap or failure.
- Use **hypervector** for HDC representations in documentation. Reserve **vector** for general mathematics or explicit comparisons with learned text/image embeddings; dimension alone does not determine the representation type.
