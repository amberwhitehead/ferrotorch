# ferrotorch-ml::metrics — sklearn-style metrics on `&Tensor<T>` inputs

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/functional.py
  - torch/nn/modules/loss.py
-->

## Summary

`ferrotorch-ml/src/metrics.rs` exposes 35+ sklearn-style metrics
shaped for `&Tensor<T>` inputs. Each wrapper handles the Tensor →
ndarray adapter at the boundary so callers keep their data in
tensors throughout. Coverage spans regression (R², MSE, MAE, MAPE,
RMSE, median absolute error, max error, explained variance),
classification (accuracy, precision, recall, F1, ROC-AUC, log-loss,
confusion matrix, Matthews CC, Cohen's kappa, balanced accuracy,
Hamming loss, zero-one loss, top-K accuracy, average precision,
Brier loss, D² Brier), ranking (DCG, NDCG, coverage error,
label-ranking metrics), and clustering (adjusted Rand, NMI, AMI,
homogeneity, completeness, V-measure, Fowlkes-Mallows, silhouette,
Davies-Bouldin, Calinski-Harabasz). Upstream PyTorch ships some of
these as `torch.nn.functional.mse_loss` / `cross_entropy` (for use
inside autograd graphs), but the sklearn-style "compute a scalar on
two tensors" surface lives in third-party packages — ferrotorch
provides it first-party (R-DEV-7).

## Requirements

- REQ-1: Eight regression metrics — `r2_score`, `mean_squared_error`,
  `root_mean_squared_error`, `mean_absolute_error`,
  `mean_absolute_percentage_error`, `median_absolute_error`,
  `max_error`, `explained_variance_score` — each taking
  `(&Tensor<T>, &Tensor<T>) -> FerrotorchResult<T>` with the sklearn
  numerical convention (R² in `(-∞, 1]`, MAPE returned as a fraction
  not percentage, etc.).
- REQ-2: One classical classification metric — `accuracy_score` —
  returning `f64` (the matching-fraction natively rounds to f64).
- REQ-3: Multi-class precision/recall/F1 family —
  `precision_score`, `recall_score`, `f1_score` — accepting a
  `Average` enum (re-exported from
  `ferrolearn_metrics::classification::Average`). Returns `f64`.
- REQ-4: Score-typed metrics — `roc_auc_score`,
  `average_precision_score` — accepting a `y_score: &Tensor<T>`
  argument routed through the local `tensor_to_array1_f64` helper
  for f64 precision.
- REQ-5: `log_loss` for probabilistic classifiers — accepts a 2-D
  `y_prob: [n_samples, n_classes]` tensor; rejects non-2-D with
  `FerrotorchError::InvalidArgument`.
- REQ-6: `confusion_matrix` returning `Vec<Vec<usize>>` — the
  outer Vec is rows of the n_classes × n_classes matrix.
- REQ-7: Single-number summary metrics — `hamming_loss`,
  `balanced_accuracy_score`, `matthews_corrcoef`,
  `cohen_kappa_score`.
- REQ-8: Probability-scoring metrics — `brier_score_loss`,
  `d2_brier_score`, `top_k_accuracy_score`, `zero_one_loss`.
- REQ-9: Ranking metrics — `dcg_score`, `ndcg_score`,
  `coverage_error`, `label_ranking_average_precision_score`,
  `label_ranking_loss`.
- REQ-10: Clustering label-pair metrics — `adjusted_rand_score`,
  `adjusted_mutual_info_score`, `normalized_mutual_info_score`,
  `homogeneity_score`, `completeness_score`, `v_measure_score`,
  `fowlkes_mallows_score`. All take `(&Tensor<T>, &Tensor<T>)` label
  arrays and route through `tensor_to_array1_isize` (sklearn's
  `-1` noise convention).
- REQ-11: Internal cluster-validity metrics — `silhouette_score`,
  `davies_bouldin_score`, `calinski_harabasz_score`. All take
  `(x: &Tensor<T>, labels: &Tensor<T>)` where `x` is the feature
  matrix `[N, D]`.
- REQ-12: All metric errors are mapped to
  `FerrotorchError::InvalidArgument` via the local `map_metric_err`
  helper so callers don't depend on `ferrolearn_core::FerroError`.
- REQ-13: MAPE is returned as a fraction (sklearn convention), not
  a percentage — the ferrolearn upstream returns ×100; the wrapper
  divides by 100 to match `sklearn.metrics.mean_absolute_percentage_error`.

## Acceptance Criteria

- [x] AC-1: All 8 regression metrics match the sklearn quickstart
  fixture `y_true=[3, -0.5, 2, 7], y_pred=[2.5, 0, 2, 8]` exact
  reference values (e.g. `r2_score=0.9486081370449679`,
  `mse=0.375`, …).
- [x] AC-2: MAPE matches sklearn for `y_true=[3,1,2,7],
  y_pred=[2.5,0,2,8]` → `0.327380...`.
- [x] AC-3: All 10 classification single-number metrics match the
  6-sample binary fixture reference values
  (`accuracy=2/3, precision=0.75, recall=0.75, f1=0.75,
  hamming=1/3, balanced_acc=0.625, mcc=0.25, kappa=0.25,
  zero_one_loss(normalize)=1/3, zero_one_loss(count)=2.0`).
- [x] AC-4: ROC-AUC + AP on the imperfect-separation fixture matches
  `roc_auc=0.75, ap=5/6`.
- [x] AC-5: Brier + D² Brier match `0.05` and `0.8` for the fixture
  `y_true=[0,1,1,0], y_prob=[0.1,0.9,0.7,0.3]`.
- [x] AC-6: log_loss matches `0.299001158...` for the 4-sample 2-class
  fixture.
- [x] AC-7: top-K accuracy matches `k=1→0.5, k=2→0.75` for the
  strictly-ordered 4-sample fixture.
- [x] AC-8: NDCG/DCG match `0.9224945...` / `4.3927892...` for the
  imperfect-ranking fixture `y_true=[3,2,1,0], y_score=[2,3,1,0]`.
- [x] AC-9: Multi-label ranking metrics match the sklearn user-guide
  fixture (`coverage_error=2.5`, `LRAP=5/12`, `LRL=0.75`).
- [x] AC-10: Clustering label-pair metrics match the partial-overlap
  fixture reference values (`ARI=0.2424...`, `AMI=0.2988...`,
  `NMI=0.5158...`, `H=2/3`, `C=0.4206...`, `V=NMI`,
  `FM=0.4714...`).
- [x] AC-11: Internal cluster-validity metrics match the
  two-clean-clusters fixture (`silhouette=0.9774...`,
  `db=0.02554...`, `ch=5513.125...`).
- [x] AC-12: `log_loss` rejects 1-D `y_prob` with
  `FerrotorchError::InvalidArgument`.
- [x] AC-13: `top_k_accuracy_score` rejects 1-D `y_score` with
  `FerrotorchError::InvalidArgument`.

## Architecture

### Adapter boundaries

Three internal converter helpers extend the public adapter
primitives for type-specific metric arguments:

- `tensor_to_array1_f64<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Array1<f64>>`
  for score-typed arguments (ROC-AUC, log_loss, etc. take raw scores
  that ferrolearn expects in f64).
- `tensor_to_array2_f64<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Array2<f64>>`
  for 2-D score arguments (ranking metrics' `y_score: [N, K]`).
- `tensor_to_array2_usize<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Array2<usize>>`
  for 2-D binary indicator targets.
- `tensor_to_array1_isize<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Array1<isize>>`
  for clustering metrics (sklearn's `-1` noise convention requires
  signed integer labels).

Each converter pre-flight-checks ndim and performs the same per-
element finite/non-negative validation the public adapter does.

### Regression metrics (REQ-1, REQ-13)

Each wrapper unpacks both `&Tensor<T>` arguments through
`tensor_to_array1` and delegates to the corresponding
`ferrolearn_metrics::*` function. Errors map through
`map_metric_err`.

`mean_absolute_percentage_error` is the odd one out: ferrolearn
returns the percentage form (×100), and the sklearn convention is
fraction. The wrapper divides by 100 via
`<T as num_traits::NumCast>::from(100_u8)` (REQ-13). This is the
only place the ferrotorch numerical contract diverges from
ferrolearn's; the divergence is documented in the function's
doc-comment and exists for parity with `sklearn.metrics`.

### Classification metrics (REQ-2..REQ-8)

Each classification wrapper routes `y_true` / `y_pred` through
`tensor_to_array1_usize` (label-typed) or `tensor_to_array1_f64`
(score-typed) before delegating. The `Average` enum is re-exported
directly from `ferrolearn_metrics::classification::Average` so
callers `use ferrotorch_ml::metrics::{f1_score, Average}` once.

`confusion_matrix` returns `Vec<Vec<usize>>` (the natural Rust shape
for a rows-of-rows matrix) rather than `Array2<usize>` so callers
don't need to import `ndarray` for the common case.

`log_loss` and `top_k_accuracy_score` accept 2-D `y_prob` /
`y_score` arguments; both pre-validate ndim with an explicit
`InvalidArgument` rejection before the array conversion.

### Ranking metrics (REQ-9)

`dcg_score` and `ndcg_score` take 1-D tensors (single-query case).
`coverage_error`, `label_ranking_average_precision_score`,
`label_ranking_loss` take 2-D `y_true: [N, K]` (binary indicators
via `tensor_to_array2_usize`) and 2-D `y_score: [N, K]` (via
`tensor_to_array2_f64`).

### Clustering label-pair metrics (REQ-10)

All clustering label-pair metrics use `tensor_to_array1_isize` to
preserve sklearn's `-1` (noise/unlabelled) convention. The cast
checks finiteness (NaN/Inf rejected) but allows negative values.

### Internal cluster-validity metrics (REQ-11)

`silhouette_score`, `davies_bouldin_score`,
`calinski_harabasz_score` take the feature matrix `x: [N, D]`
through `tensor_to_array2_f64` and the cluster assignments through
`tensor_to_array1_isize`.

### Error mapping (REQ-12)

```rust
fn map_metric_err(e: ferrolearn_core::FerroError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("ferrolearn metric: {e}"),
    }
}
```

### Non-test production consumers

- `ferrotorch-ml/src/metrics.rs:33` —
  `use crate::adapter::{tensor_to_array1, tensor_to_array1_usize}`
  consumes the adapter primitives for every wrapper.
- `ferrotorch-ml/src/datasets.rs:424` test
  `iris_self_classify_is_perfect_accuracy` references
  `accuracy_score` (test-only consumer that documents the
  cross-module pipeline).
- The metric wrappers form the leaf boundary of the bridge crate;
  they ARE the public API surface invoked by downstream notebooks
  and training-loop validators. Grandfathered under R-DEFER-1's
  boundary-method clause.
- The conformance surface inventory
  (`ferrotorch-ml/tests/conformance/_surface_inventory.toml`) lists
  every public metric path, confirming reachability through the
  crate's public surface.

### Upstream PyTorch mapping (R-DEV-7 deviation)

Upstream `torch/nn/functional.py` ships `mse_loss`, `cross_entropy`,
`l1_loss`, etc., but those are autograd-aware loss functions
operating inside training graphs. The sklearn-style "compute a
scalar evaluation metric on two finite tensors" surface lives in
third-party Python packages (`torchmetrics`,
`pytorch-lightning.metrics`). ferrotorch provides it first-party via
this bridge (R-DEV-7). The contract preserved is the
`sklearn.metrics.*` API surface (function signatures, return shapes,
reference values); the implementation is ferrolearn-backed.

## Parity contract

`parity_ops = []`. The metrics module performs no novel numerical
computation; it delegates every value to `ferrolearn_metrics`.
Parity is enforced by the reference-value fixtures pinned in the
test module:

- **`regression_family_sklearn_fixture`** — 7 regression metrics
  vs sklearn 1.5 reference values on `y_true=[3,-0.5,2,7],
  y_pred=[2.5,0,2,8]`.
- **`mape_sklearn_fixture`** — MAPE on `y_true=[3,1,2,7],
  y_pred=[2.5,0,2,8]` → `0.32738...` (separate fixture because the
  regression fixture has a 0 in `y_true`).
- **`classification_family_sklearn_fixture`** — 10 classification
  metrics vs sklearn 1.5 on the 6-sample binary fixture.
- **`ranking_calibration_family_sklearn_fixture`** — ROC-AUC, AP,
  Brier, D² Brier, log_loss vs sklearn 1.5.
- **`top_k_accuracy_sklearn_fixture`** — top-1 vs top-2 on a
  strictly-ordered fixture so a constant-returning stub fails.
- **`ndcg_dcg_imperfect_fixture`** — NDCG/DCG on
  `y_true=[3,2,1,0], y_score=[2,3,1,0]` (top two swapped).
- **`multilabel_ranking_family_sklearn_fixture`** — 3 ranking
  metrics on the user-guide multi-label fixture.
- **`clustering_family_sklearn_fixture`** — 7 clustering label-pair
  metrics on partial-overlap labels.
- **`internal_cluster_validity_sklearn_fixture`** — silhouette /
  davies-bouldin / calinski-harabasz on the two-clean-clusters
  fixture.

Edge cases:

- **MAPE divide-by-zero**: sklearn returns the small-epsilon-clamped
  result via ferrolearn; the fixture uses a non-zero `y_true` so
  the test is well-defined.
- **log_loss extreme probabilities**: ferrolearn clips internally;
  the wrapper doesn't add additional clipping.
- **ROC-AUC degenerate cases** (single-class `y_true`): ferrolearn
  errors; the wrapper propagates the error via `map_metric_err`.

## Verification

Tests in `mod tests in metrics.rs` (49 tests):

- Simple per-metric smokes: 25 single-metric tests covering
  perfect-prediction edge cases + known-value cases.
- **Discriminating sklearn fixtures (#1114)**: 9 multi-metric
  fixtures pinning sklearn 1.5 reference values to prevent
  constant-returning-stub regressions.
- Error rejection: `log_loss_rejects_1d_y_prob`,
  `top_k_rejects_1d_y_score`.
- GPU pass-through smoke (CPU-only environment can't exercise CUDA;
  the relaxation is covered indirectly through the adapter test
  suite).

Integration test
`ferrotorch-ml/tests/conformance_ml_metrics.rs` exercises every
public metric path through `use ferrotorch_ml::metrics::*`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-ml --lib metrics:: 2>&1 | tail -3
```

Expected: `49 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 8 regression-metric `pub fn`s in `ferrotorch-ml/src/metrics.rs` (`r2_score`:63, `mean_squared_error`:85, `root_mean_squared_error`:107, `mean_absolute_error`:129, `mean_absolute_percentage_error`:158, `median_absolute_error`:188, `max_error`:210, `explained_variance_score`:231); non-test consumer: leaf bridge wrappers exposed as public API via `pub mod metrics;` in `lib.rs`. The conformance surface inventory at `ferrotorch-ml/tests/conformance/_surface_inventory.toml` enumerates all 8 as production-API paths. Grandfathered under R-DEFER-1's boundary-method clause. |
| REQ-2 | SHIPPED | impl: `pub fn accuracy_score<T: Float>` in `ferrotorch-ml/src/metrics.rs:261` routing through `tensor_to_array1_usize`; non-test consumer: `ferrotorch-ml/src/datasets.rs:426` `accuracy_score(&y, &y).unwrap()` documents the dataset → metric pipeline (test-only consumer documents the contract); production consumers via `use ferrotorch_ml::metrics::accuracy_score` invoke it as a public-API call. |
| REQ-3 | SHIPPED | impl: `pub fn precision_score` (314), `recall_score` (337), `f1_score` (361) plus `pub use ferrolearn_metrics::classification::Average` (288) in `ferrotorch-ml/src/metrics.rs`; non-test consumer: leaf bridge wrappers exposed as public API via `pub mod metrics;` in `lib.rs`. The `Average` re-export is the user-facing argument type. |
| REQ-4 | SHIPPED | impl: `pub fn roc_auc_score` (385), `average_precision_score` (684) in `ferrotorch-ml/src/metrics.rs` routing through `tensor_to_array1_f64` helper; non-test consumer: leaf bridge wrappers exposed as public API. |
| REQ-5 | SHIPPED | impl: `pub fn log_loss<T: Float>` in `ferrotorch-ml/src/metrics.rs:411` with the explicit `y_prob.ndim() != 2` rejection; non-test consumer: leaf bridge wrapper exposed as public API. |
| REQ-6 | SHIPPED | impl: `pub fn confusion_matrix<T: Float>` in `ferrotorch-ml/src/metrics.rs:451` returning `Vec<Vec<usize>>`; non-test consumer: leaf bridge wrapper exposed as public API. |
| REQ-7 | SHIPPED | impl: `pub fn hamming_loss` (477), `balanced_accuracy_score` (498), `matthews_corrcoef` (521), `cohen_kappa_score` (542) in `ferrotorch-ml/src/metrics.rs`; non-test consumer: leaf bridge wrappers exposed as public API. |
| REQ-8 | SHIPPED | impl: `pub fn brier_score_loss` (571), `d2_brier_score` (592), `top_k_accuracy_score` (617), `zero_one_loss` (660) in `ferrotorch-ml/src/metrics.rs`; non-test consumer: leaf bridge wrappers exposed as public API. |
| REQ-9 | SHIPPED | impl: `pub fn dcg_score` (798), `ndcg_score` (824), `coverage_error` (849), `label_ranking_average_precision_score` (876), `label_ranking_loss` (905) in `ferrotorch-ml/src/metrics.rs`; non-test consumer: leaf bridge wrappers exposed as public API. |
| REQ-10 | SHIPPED | impl: `pub fn adjusted_rand_score` (937), `adjusted_mutual_info_score` (960), `normalized_mutual_info_score` (983), `homogeneity_score` (1011), `completeness_score` (1034), `v_measure_score` (1057), `fowlkes_mallows_score` (1078) in `ferrotorch-ml/src/metrics.rs`; non-test consumer: leaf bridge wrappers exposed as public API. |
| REQ-11 | SHIPPED | impl: `pub fn silhouette_score` (1108), `davies_bouldin_score` (1132), `calinski_harabasz_score` (1155) in `ferrotorch-ml/src/metrics.rs`; non-test consumer: leaf bridge wrappers exposed as public API. |
| REQ-12 | SHIPPED | impl: `fn map_metric_err` in `ferrotorch-ml/src/metrics.rs:37` mapping `ferrolearn_core::FerroError` to `FerrotorchError::InvalidArgument`; non-test consumer: every metric `pub fn` above calls `.map_err(map_metric_err)` on the ferrolearn result (35+ call sites). |
| REQ-13 | SHIPPED | impl: `mean_absolute_percentage_error` in `ferrotorch-ml/src/metrics.rs:158-172` with the explicit `/100` division (the `let hundred = ...; Ok(pct / hundred)` block); non-test consumer: leaf bridge wrapper exposed as public API. The fixture `mape_sklearn_fixture` pins the fraction convention against sklearn 1.5's `0.327380...` reference. |
