# Agent-driven compression pilot

This directory is a runnable first experiment: agents propose **occurrence
priorities**, and a fixed evaluator measures their effect on graph compression.
It is a synthetic development benchmark, not a new compression result on ADMET,
REDDIT, or molecular data. No API key, GPU, or agent framework is needed to
reproduce the included candidate evaluations.

## Run it

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[tests]'
python research/agent_compression/evaluate.py \
  --output artifacts/agent_compression/dev.json
```

The default compares the original boundary-first ranking with three proposals:
largest motif first, structural yield per boundary edge, and conflict-aware
packing. Each run fits the same training corpus once, then uses the same frozen
prefix of up to three rules for every proposal. A separate positive control must select
real rewrites and achieve positive MDL savings.

The JSON records source hashes, the base Git revision, rule identities,
reconstruction checks, codelength components, and elapsed time. The two reports
serve different purposes:

- `forced_dictionary_diagnostic`: applies every eligible rewrite using the fixed
  dictionary. This exposes packing differences even when the code becomes
  **longer**. Lower `encoded_bits` is better, but negative savings remain a loss.
- `sparse_selector`: applies the unchanged corpus selector to those same
  proposed rewrites. The selector may keep every graph unchanged. The one-bit
  model-choice charge can yield savings of -1 bit in that case.

`unmodified_pipeline` also reports the ordinary compressor, using the dictionary
it selected during training. This can differ from the fixed three-rule packing
experiment. A better fixed-dictionary ranking is not automatically an
end-to-end improvement over the ordinary compressor.

The checked-in [development_result.json](development_result.json) records the
first working development round. To run the evaluator checks and related
compression checks:

```bash
python -m pytest -q research/agent_compression/test_evaluate.py \
  tests/test_compression_public_api.py tests/test_mdl.py \
  tests/test_mdl_boundary_bits_overflow.py
```

## Launch agents

In ChatGPT Work or a Codex session opened on this repository, ask for parallel
subagents directly. The coordinator starts the workers and collects their
results. A useful first-round prompt is:

```text
Run one bounded agent-search round in research/agent_compression.

First read its README and freeze the evaluator and dataset source hashes.
Spawn three proposal agents in parallel. Give them different hypotheses:
1. improve the tradeoff between contraction size and boundary cost;
2. account for conflicts between overlapping occurrences;
3. find a simple alternative that challenges the existing ranking.

Each worker may write only its own files under
research/agent_compression/proposals/, exposing priority(features).
Each may submit at most two proposals. Keep the current model and reasoning
settings the same across workers. Do not edit evaluate.py, datasets.py,
candidates.py, src/, or tests/. Do not use final-test data.

The coordinator evaluates the fixed baseline and at most six proposals on
development data, reviews the code and reconstruction checks, and records
every success and failure. Compare estimated MDL bits and elapsed time.
Stop after this round and report the best candidate, its hash, and limitations.
Do not merge, publish results, or start an unattended loop.
```

When writing in parallel, give workers separate files or separate Git worktrees.
Here the proposal filenames provide ownership; the coordinator owns the
evaluator. Starting more agents does not itself make a scientific result.
Later, compare one worker and three workers at the **same total proposal and
model-call budgets** before claiming a benefit from parallel agent search.

Official launch instructions:

- https://learn.chatgpt.com/docs/agent-configuration/subagents
- https://learn.chatgpt.com/docs/environments/git-worktrees

## Candidate interface

An agent file must expose one function. For example:

```python
def priority(features):
    k = features["motif_nodes"]
    b = features["boundary_edges"]
    return (-(k - 1) / (1 + b), b)
```

Features are numeric `motif_nodes`, `motif_edges`, `boundary_edges`, and
`conflicts`. Return a nonempty tuple of finite numbers; lower tuples are chosen
first. The evaluator owns stable tie-breaking, greedy nonoverlap selection,
contraction, decoding, and MDL scoring. NaN, infinity, exceptions, and malformed
outputs invalidate the proposal.

Evaluate an individual proposal with:

```bash
python research/agent_compression/evaluate.py \
  --candidate baseline \
  --candidate research/agent_compression/proposals/agent_1.py \
  --output artifacts/agent_compression/round_1.json
```

These are trusted, reviewed Python proposals running in the evaluator process.
Immutable feature dictionaries define the interface; they do not sandbox Python
or prevent a malicious import from changing the process. A future unattended
runner needs process isolation, timeouts, restricted file access, and independent
parent-side verification. This starter does not launch paid API requests or
continue agent work after a chat turn ends.

## Final evaluation

Only choose a final seed after selecting and freezing one proposal using the
development results. Keep that seed outside proposal-agent prompts. Supply it
in `FINAL_SEED`, then run, for example:

```bash
python research/agent_compression/evaluate.py \
  --phase test \
  --candidate conflict_aware \
  --freeze-from artifacts/agent_compression/dev.json \
  --evaluation-seed "${FINAL_SEED:?Choose a fresh final seed first}" \
  --output artifacts/agent_compression/final.json
```

Use the selected proposal's actual name or path. The command checks its source
hash against development, checks evaluator/dataset/baseline hashes, requires the
same corpus size and a new seed, and evaluates the chosen proposal **plus the
fixed baseline**. It does not sweep other proposals on the final graphs.

This is a workflow convention, not a security boundary: local agents with file
access can inspect code. Distinct generator seeds also do not guarantee
nonisomorphic graphs. Replace this synthetic corpus with a versioned real-data
split and check duplicate structures before making generalization claims.

## What remains fixed

- Candidate motifs and the dictionary come from training only.
- Topology-only simple undirected graphs are the initial coding scope.
- Every proposed rewrite is decoded through the public decoder, including
  rewrites rejected by the sparse selector, and compared up to isomorphism.
- Reported bits include graph/template, boundary, dictionary, selector, and
  model-choice accounting. These are analytical MDL estimates, not emitted
  archive bytes.
- All compressor and decoder implementation under `src/` remains unchanged.

The next scientific step is to choose a real corpus with an informative baseline,
repeat development across multiple seeds, and compare equal search budgets.
