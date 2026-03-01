# Decision Rubric

## Candidate Eligibility

Treat tests as redundancy candidates only if cluster size is greater than 1 for `(entrypoint, intent)`.

Parametrized tests (`@pytest.mark.parametrize`) are never auto-delete/auto-merge outcomes and always
receive `KEEP_FOR_SIGNAL`; their individual parameter variants must be assessed separately.

## Evidence Sources

1. Empirical deselection run (`pytest --deselect=<nodeid>` against target suites)
2. Structural dominance: peer assertion type set is a superset (including equal) of the candidate's
3. Source token Jaccard similarity: normalised token overlap between candidate and each cluster peer
4. Test name similarity: `difflib.SequenceMatcher` ratio between test function names in the cluster
5. Assertion count: number of explicit `assert` statements in the function body
6. Unique-signal features (from ranked report when available): unique lines, branches, mutants
7. Cross-suite overlap ratio (from ranked report when available)
8. Live coverage fallback (when ranked coverage is absent): per-test line/branch uniqueness and optional cross-suite overlap from `--comparator-suite`
9. Branch-equivalence signal: candidate vs nearest retained peer branch-token parity (exact match, Jaccard, candidate-only/anchor-only deltas)
10. Hardened confidence gates (repo-agnostic): deselection, strict stability, branch parity/similarity, dominance, mutation uniqueness, coverage uniqueness, overlap, plus explicit external-manual gates

When ranked mutation data is unavailable for a run, rely on deselection + structural
signals and treat delete recommendations as lower confidence until mutation signal is supplied.
When ranked coverage data is unavailable, the script attempts to generate coverage signal directly.
When branch-token sets are unavailable from prior coverage artifacts, the script re-collects
per-test branch coverage for candidate/anchor pairs to keep branch-equivalence evidence present.
External gates (`side_effects`, `nonfunctional`, `history_regression`) are intentionally marked
`unknown` by automation and require manual signoff if you want maximal confidence.

## Default Thresholds

- `high_overlap_threshold`: 0.97
- `low_unique_signal`: unique lines = 0, unique branches = 0, unique mutants = 0
- `src_similarity_delete`: ≥ 0.90 (near-duplicate source; combined with dominated/low-signal)
- `src_similarity_merge`: ≥ 0.80 (structurally similar source)
- `name_similarity_merge`: ≥ 0.85 (test names suggest copy-variant pattern)
- `delete_requires`: deselection pass + peer superset + low unique signal + high overlap
- `delete_alt_requires`: deselection pass + src_similarity ≥ 0.90 + (dominated OR low_unique_signal)

## Conservative Overrides

Promote to keep when any applies:

- cluster would become empty
- deselection run fails (`KEEP_FOR_STABILITY`)
- candidate is a parametrized test function
- candidate is only test asserting mutability/readonly contract in cluster
- candidate is only test with explicit exception semantics in cluster

## Batch Deletion Safeguard

After per-test scoring, if all tests in a `(entrypoint, intent)` cluster are `DELETE_SAFE_HIGH`,
retain one anchor as `KEEP_FOR_CONTRACT` to avoid full cluster deletion in one batch.

## Strict Delete Gate (Optional but Recommended Before Real Prunes)

When `--strict-delete-gate` is enabled, a candidate tagged `DELETE_SAFE_HIGH` must also pass:

1. Repeated deselection stability (`--strict-repeats`, default 3)
2. Staged batch simulation (`--strict-batch-size`)
3. Mandatory post-suite pass (`--strict-post-suite`, default `tests` when present)
4. Mutation-probe kill delta gate:
   - baseline kill count from lightweight probes
   - batch kill count with deselected candidates
   - must satisfy `batch_kills >= baseline_kills - strict_mutation_max_drop`
   - all selected probes must apply successfully in baseline and batch runs
5. Re-audit pass after each accepted batch

Any delete candidate failing strict gate is downgraded to keep.

## Decision Logic (evaluated top-to-bottom)

1. `is_parametrized` → `KEEP_FOR_SIGNAL`
2. deselection fails → `KEEP_FOR_STABILITY`
3. unique mutability/readonly or exception semantics in cluster → `KEEP_FOR_SIGNAL`
4. `low_signal AND dominated` → `DELETE_SAFE_HIGH`
5. `src_similarity ≥ 0.90 AND (low_signal OR dominated)` → `DELETE_SAFE_HIGH`
6. `dominated OR src_similarity ≥ 0.80 OR name_similarity ≥ 0.85` → `MERGE_RECOMMENDED`
7. otherwise → `KEEP_FOR_SIGNAL`
8. batch safeguard pass: if a cluster is all-delete, keep one anchor as `KEEP_FOR_CONTRACT`

## Merge Guidance

If candidate is not delete-safe but deselection passes:

- recommend merge into nearest cluster peer
- preserve strongest assertion surface from both tests
- prefer parametrized merged tests for input-variant duplicates

## Hardened Confidence Tiers

Automation emits `confidence_tier` for each row:

- `GOLD_DELETE_CANDIDATE`
  - decision is delete-safe
  - strict gate passed
  - branch exact-equivalence passed
  - dominance passed
  - unique mutants = 0
  - unique coverage signal = 0
  - overlap threshold passed
- `SILVER_DELETE_CANDIDATE`
  - decision is delete-safe
  - measured core gates pass
  - branch gate is high similarity (or strict gate unavailable)
- `BRONZE_DELETE_REVIEW`
  - delete-safe label exists, but one or more measured gates failed
- `MERGE_CANDIDATE`
  - non-delete merge recommendation
- `KEEP_CANDIDATE`
  - keep recommendation (signal/contract/stability)

Use `GOLD` as default auto-delete threshold.
Use `SILVER` only with manual signoff on external gates and domain context.
Treat `BRONZE` as refactor/merge-first, not immediate delete.

## Output Interpretation

- `DELETE_SAFE_HIGH`: immediate prune candidate
- `MERGE_RECOMMENDED`: refactor candidate
- `KEEP_FOR_SIGNAL`: keep; distinct source, assertion, or structural signal
- `KEEP_FOR_CONTRACT`: keep; required to retain minimal cluster contract
- `KEEP_FOR_STABILITY`: keep; operationally required right now
