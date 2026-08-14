# Curated notebooks

These notebooks are package-backed entry points rather than copies of the
historical exploratory implementations. Large datasets and generated outputs
are not committed.

- `chemistry/mdl_chemistry_end_to_end.ipynb`: labeled molecular graphs, exact
  decoding, and model-agnostic compressed graphs.
- `reddit/reddit_mdl_diagnostics.ipynb`: honest and forced-diagnostic MDL runs
  on an external TU-format REDDIT-MULTI-5K dataset.
- `syntax/mdl_ast_negative_result.ipynb`: dependency-tree-style graphs and a
  negative-result workflow.

Install notebook dependencies with:

```bash
python -m pip install -e ".[notebooks]"
```

The full historical notebooks remain recoverable from the pre-cleanup `main`
history and should not be copied back into the production branch.
