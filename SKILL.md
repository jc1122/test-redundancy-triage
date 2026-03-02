---
name: test-redundancy-triage
description: >
  Identify and safely remove redundant tests using empirical deselection,
  branch-equivalence analysis, assertion dominance, and coverage/mutation
  signals. Classifies each test as DELETE, MERGE, or KEEP with confidence
  tiers (Gold/Silver/Bronze). Includes strict delete gates and cluster
  safeguards to prevent accidental coverage loss.
---

# Test Redundancy Triage

## Overview
Evaluate test redundancy with a conservative, evidence-based workflow. Prefer merge recommendations unless a candidate is explicitly validated as high-confidence delete.

### Heuristics applied
- **Empirical deselection**: `pytest --deselect=<nodeid>` confirms the suite still passes without the candidate.
- **Structural dominance**: a peer's assertion-type set is a superset (or equal) of the candidate's.
- **Source token Jaccard similarity**: normalised token overlap ≥ 0.80 flags structural near-duplicates; ≥ 0.90 combined with dominance/low-signal escalates to `DELETE_SAFE_HIGH`.
- **Test name similarity**: `difflib.SequenceMatcher` ratio ≥ 0.85 within a cluster flags copy-variant candidates for merge.
- **Assertion count**: raw `assert` statement count distinguishes a 5-assert test from a 1-assert test even when both share the same assertion type label.
- **Parametrized test detection**: `@pytest.mark.parametrize` tests are always emitted as `KEEP_FOR_SIGNAL` (never delete/merge automatically); they require per-variant analysis.
- **Coverage/mutation data** (when ranked CSV is available): unique lines, branches, and mutants killed give the strongest delete signal.
- **Live coverage fallback**: when ranked coverage is absent, the script collects per-test line/branch signal with `coverage.py` (and bootstraps `coverage` into a runtime target path if needed).
- **Branch-equivalence evidence**: candidate-to-anchor branch-token comparison (exact match + Jaccard + deltas) is generated as explicit redundancy evidence.
- **Hardened confidence gates**: each candidate is evaluated against explicit gates (deselect, strict stability, branch parity/similarity, dominance, mutation uniqueness, coverage uniqueness, overlap) and assigned a confidence tier.
- **Cluster anchor safeguard**: if all tests in one `(file, intent)` cluster are flagged delete-safe, one anchor is auto-retained as `KEEP_FOR_CONTRACT`.
- **Strict delete gate (optional)**: repeated deselection, staged batch simulation, post-suite gate, and mutation-probe delta checks before finalizing delete-safe decisions.

## Prerequisites
- Python 3.10+
- `pytest` (required)
- `coverage` / `coverage.py` (optional; enables live branch/line signal when no ranked CSV is provided — the script will attempt to bootstrap it if absent)
- `pytest-xdist` (optional; speeds up parallel deselection runs)
- `numba` (optional; only relevant if your tests import it). The script does **not** stub numba by default; use `--allow-numba-stub` only when you intentionally want fallback stubbing.

## Run Workflow
1. Choose target suite files.
2. Run bundled script `scripts/triage_redundancy.py`.
3. Review `validation_decision` for each candidate.
4. Propose edits only for `DELETE_SAFE_HIGH` first.

Use this command pattern:

```bash
python scripts/triage_redundancy.py \
  --root /path/to/repo \
  --python /path/to/python \
  --suite tests/test_api.py \
  --suite tests/test_core.py \
  --comparator-suite tests/test_integration.py \
  --source-prefix src/mypackage/ \
  --out-dir artifacts/redundancy \
  --max-workers 4
```

`--suite` (required) specifies the test files to evaluate; pass it multiple times for multiple files.
All suite paths (`--suite`, `--comparator-suite`, `--strict-post-suite`) must exist under `--root`.
`--comparator-suite` (optional) adds extra test files used only for cross-suite overlap scoring, not as candidates.
`--source-prefix` (optional) restricts coverage token collection to files under that path prefix (e.g. `src/mypackage/`); omit to collect from all source files.
If `--ranked-csv` and `--inventory-csv` are missing, the script still performs empirical deselection checks and gathers live coverage signal when possible.
The script accepts either `--python python3` (PATH lookup) or a path like `--python .venv/bin/python`.
The runtime environment auto-detects conventional import roots: repo root and `src/` (if present),
so it works for both flat and `src` package layouts without extra PYTHONPATH setup.

Strict mode (recommended before actual deletions):

```bash
python scripts/triage_redundancy.py \
  --root /path/to/repo \
  --python python3 \
  --suite tests/test_api.py \
  --comparator-suite tests/test_integration.py \
  --source-prefix src/mypackage/ \
  --strict-delete-gate \
  --strict-post-suite tests \
  --strict-repeats 3 \
  --strict-batch-size 8 \
  --strict-mutation-probes 3 \
  --strict-mutation-max-drop 0 \
  --mutation-probes-config path/to/probes.json
  # --allow-numba-stub   # optional: only if your suite requires numba fallback
```

## Parallelism Note

The triage script and TQA audit script are fully independent and can be run concurrently on separate terminals. For an orchestrated pipeline, see the `test-audit-pipeline` skill.

## Custom Mutation Probes

The strict delete gate uses mutation probes to verify that the test suite detects injected faults.
No built-in probes are provided — supply them via `--mutation-probes-config` with a JSON file:

```json
[
  {
    "probe_id": "P001",
    "file": "src/mypackage/core.py",
    "old": "return compute_result(x, y)",
    "new": "return None"
  },
  {
    "probe_id": "P002",
    "file": "src/mypackage/validators.py",
    "old": "if value < 0:",
    "new": "if False and value < 0:"
  }
]
```

Each probe does a **single exact-string replacement** in the given file (relative to `--root`).
The script errors cleanly if the `old` string appears zero or more than once in that file.
`--strict-mutation-probes N` controls how many probes from the list are used (default 3).
Strict mode requires **all selected probes** to apply successfully in baseline and batch runs.
If any selected probe cannot be applied, the mutation gate fails and delete candidates are downgraded.
When no `--mutation-probes-config` is provided, the mutation gate is skipped.

## Clustering and Entrypoint Inference

Tests are grouped into `(entrypoint, intent)` clusters; redundancy is evaluated only within a cluster.
By default the entrypoint is `test_file_path::test_class` (or `::<module>` for module-level tests).
This intentionally keeps clustering broad enough to surface redundancy candidates in API-style suites.
The `infer_entrypoint()` function in the script can be extended with project-specific API name checks
(e.g. matching function call names from `calls` or patterns in `src`) to produce finer-grained clusters.
Intent is inferred generically from assertion types and test name keywords (`version` → introspection,
`exception` assertions → error_semantics, etc.).

## Decision Policy
- `DELETE_SAFE_HIGH`: delete candidate now; deselection passed and test is dominated by peer assertions with low unique signal.
- `MERGE_RECOMMENDED`: do not hard-delete yet; consolidate with similar test(s) into one stronger parametrized test.
- `KEEP_FOR_SIGNAL`: keep; candidate still contributes meaningful distinct checks.
- `KEEP_FOR_CONTRACT`: keep; removing would leave no test in that file+intent cluster.
- `KEEP_FOR_STABILITY`: keep; suite fails when deselected.

Detailed scoring and thresholds are in `references/decision-rubric.md`.

## Concurrency Safety
Always run candidate checks with isolated caches and without nested xdist:
- use `-n 0` only when `pytest-xdist` is available
- force clean addopts (`-o addopts=`)
- per-process `cache_dir` temp path
- bounded workers (default 4)

The bundled script already enforces this.

## Outputs
Expect these files in `--out-dir`:
- `inventory.csv`
- `coverage_matrix.csv`
- `coverage_summary.json`
- `mutation_matrix.csv`
- `mutation_summary.json`
- `branch_equiv_report.csv`
- `branch_equiv_summary.json`
- `branch_equiv_report.md`
- `confidence_gate_matrix.csv`
- `candidate_validation.csv`
- `candidate_validation.md`
- `candidate_validation_summary.json`
- `strict_gate.csv` (only with `--strict-delete-gate`)
- `strict_gate_summary.json` (only with `--strict-delete-gate`)

`mutation_matrix.csv` is always generated. If ranked mutation signal is missing for a test,
the file marks `mutation_signal_available=False` with an explanatory `status_note`.
`coverage_matrix.csv` is always generated. If ranked coverage signal is missing, the script
collects live coverage; if coverage cannot be provisioned, rows are still emitted with
`coverage_signal_available=False` and an explanatory `status_note`.
`branch_equiv_report.csv` is always generated for deselection-pass candidates paired to their
nearest in-cluster anchor; when coverage tokens are missing, the script performs live per-test
coverage collection and records availability in `branch_equiv_summary.json`.
`confidence_gate_matrix.csv` is always generated and includes hardened gate statuses plus
`confidence_tier`:
- `GOLD_DELETE_CANDIDATE`: all measured gates pass with exact branch-equivalence
- `SILVER_DELETE_CANDIDATE`: all measured core gates pass, but branch gate is high-similarity (or strict gate unavailable)
- `BRONZE_DELETE_REVIEW`: delete candidate with one or more failed measured gates
- `MERGE_CANDIDATE` / `KEEP_CANDIDATE`: non-delete outcomes

External gates (`side_effects`, `nonfunctional`, `history_regression`) are intentionally marked
`unknown` for manual signoff in a repo-agnostic workflow.
In strict mode, only `DELETE_SAFE_HIGH` candidates that pass strict gates remain deletions;
others are automatically downgraded to keep decisions with strict failure notes.
`strict_gate.csv` includes explicit mutation gate diagnostics:
`mutation_baseline_status`, `mutation_batch_status`, `mutation_applied_probes`,
and `mutation_failed_to_apply`.
Use `candidate_validation.csv` as source of truth for automated change planning.

## Known Limitations

- **Mock-heavy tests**: Tests using monkeypatch, ctypes stubs, or similar mocking may be flagged as low-signal due to assertion-type dominance overshadowing branch-level uniqueness. Always cross-check DELETE candidates that mock internal state.
- **Keyword-based intent inference**: Intent inference is keyword-based and may misclassify tests that don't follow naming conventions. Custom intent rules can be added to `infer_intent()` for project-specific patterns.
- **Parametrized test conservatism**: Parametrized tests are conservatively kept (never auto-deleted). After merging tests into parametrized form, re-run triage to clear stale classifications.
- **No per-variant analysis**: The script does not support per-variant analysis of parametrized tests. All variants are treated as a single test for coverage and branch equivalence.
- **Environment variables**: Environment variables needed by the test suite (e.g., `NUMBA_DISABLE_JIT=1`) must be passed via `--env` or set in the calling shell.
