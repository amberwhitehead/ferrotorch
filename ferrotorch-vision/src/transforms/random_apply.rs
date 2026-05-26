// CL-332: Vision Transforms & Augmentation — RandomApply / RandomChoice
//! ## REQ status (per `.design/ferrotorch-vision/transforms/random_apply.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct RandomApply<T: Float>` with `transforms: Vec<Box<dyn Transform<T>>>` and `p: f64` in `random_apply.rs`, mirroring `torchvision/transforms/v2/_container.py:63` `class RandomApply`; consumer: `pub use random_apply::{RandomApply, RandomChoice};` in `mod.rs` and `RandomApply` in the crate-root re-export in `lib.rs`. |
//! | REQ-2 | SHIPPED | `pub fn RandomApply::new(transforms, p) -> FerrotorchResult<Self>` constructor with range check in `random_apply.rs`; consumer: registered in `tests/conformance/_surface_inventory.toml` as `ferrotorch_vision::RandomApply::new`; reachable through the crate-root re-export. |
//! | REQ-3 | SHIPPED | `impl<T: Float> Transform<T> for RandomApply<T>` with random gate and chained-apply loop in `random_apply.rs`; consumer: any `Box<dyn Transform<T>>` slot — composes into nested `Compose` / `RandomApply` pipelines. |
//! | REQ-4 | SHIPPED | `pub struct RandomChoice<T: Float>` with `transforms: Vec<Box<dyn Transform<T>>>` in `random_apply.rs`, mirroring `torchvision/transforms/v2/_container.py:119` `class RandomChoice`; consumer: same `pub use` in `mod.rs` and `RandomChoice` in the crate-root re-export in `lib.rs`. |
//! | REQ-5 | SHIPPED | `pub fn RandomChoice::new(transforms) -> FerrotorchResult<Self>` constructor with non-empty check in `random_apply.rs`; consumer: registered in `tests/conformance/_surface_inventory.toml` as `ferrotorch_vision::RandomChoice::new`; reachable through the crate-root re-export. |
//! | REQ-6 | SHIPPED | `impl<T: Float> Transform<T> for RandomChoice<T>` with uniform index sampling and `.min(n - 1)` clamp in `random_apply.rs`; consumer: any `Box<dyn Transform<T>>` slot via the crate-root re-export. |
//! | REQ-7 | SHIPPED | `RandomChoice::with_p(Vec<f64>)` builder + cumulative-weight sampling in `apply` in `random_apply.rs`; consumer: reachable through the crate-root re-export — augmentation pipelines call `RandomChoice::new(ts)?.with_p(vec![0.5, 0.25, 0.25])?` to express a weighted choice per upstream `_container.py:138-141`. |

use super::rng::random_f64;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_data::Transform;

/// Apply a list of transforms sequentially with probability `p`.
///
/// With probability `p`, all contained transforms are applied in order
/// (like [`Compose`]). With probability `1 - p`, the input is returned
/// unchanged.
///
/// This mirrors `torchvision.transforms.RandomApply`.
pub struct RandomApply<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
    p: f64,
}

impl<T: Float> RandomApply<T> {
    /// Create a new `RandomApply`.
    ///
    /// * `transforms` — the transforms to apply when triggered.
    /// * `p` — probability that the transforms are applied. Must be in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `p` is outside `[0, 1]`.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>, p: f64) -> FerrotorchResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("RandomApply: p must be in [0.0, 1.0], got {p}"),
            });
        }
        Ok(Self { transforms, p })
    }
}

impl<T: Float> Transform<T> for RandomApply<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if random_f64() >= self.p {
            return Ok(input);
        }
        let mut current = input;
        for t in &self.transforms {
            current = t.apply(current)?;
        }
        Ok(current)
    }
}

/// Randomly pick one transform from a list and apply it.
///
/// Each contained transform has equal probability `1/n` of being selected.
///
/// This mirrors `torchvision.transforms.RandomChoice`.
pub struct RandomChoice<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
    /// Optional per-transform weights, parallel to `transforms`. When `None`,
    /// selection is uniform. When `Some(ws)`, each `ws[i]` is the
    /// (non-normalized) weight of the corresponding transform; cumulative
    /// sampling picks the first index whose cumulative weight exceeds the
    /// scaled uniform draw. Mirrors upstream
    /// `torchvision.transforms.v2.RandomChoice(transforms, p=...)`.
    weights: Option<Vec<f64>>,
}

impl<T: Float> RandomChoice<T> {
    /// Create a new `RandomChoice`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `transforms` is empty.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> FerrotorchResult<Self> {
        if transforms.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "RandomChoice: transforms list must not be empty".into(),
            });
        }
        Ok(Self {
            transforms,
            weights: None,
        })
    }

    /// Attach a non-uniform weight vector. `p.len()` must equal the number
    /// of transforms; weights must be finite, non-negative, and not all
    /// zero. Mirrors upstream's `p: list[float]` arg
    /// (`torchvision/transforms/v2/_container.py:138-141`).
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the length mismatches
    /// or weights are invalid.
    pub fn with_p(mut self, p: Vec<f64>) -> FerrotorchResult<Self> {
        if p.len() != self.transforms.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomChoice::with_p: expected {} weights, got {}",
                    self.transforms.len(),
                    p.len()
                ),
            });
        }
        let mut sum = 0.0;
        for &w in &p {
            if !w.is_finite() || w < 0.0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "RandomChoice::with_p: weight must be finite and >= 0, got {w}"
                    ),
                });
            }
            sum += w;
        }
        if sum <= 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RandomChoice::with_p: weights must not all be zero".into(),
            });
        }
        self.weights = Some(p);
        Ok(self)
    }
}

impl<T: Float> Transform<T> for RandomChoice<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let n = self.transforms.len();
        let idx = match &self.weights {
            None => {
                let i = (random_f64() * n as f64) as usize;
                i.min(n - 1) // Clamp in case random_f64() yields exactly 1.0.
            }
            Some(weights) => {
                let total: f64 = weights.iter().sum();
                let mut draw = random_f64() * total;
                let mut chosen = n - 1;
                for (i, &w) in weights.iter().enumerate() {
                    if draw < w {
                        chosen = i;
                        break;
                    }
                    draw -= w;
                }
                chosen
            }
        };
        self.transforms[idx].apply(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;
    use ferrotorch_data::Normalize;

    #[test]
    fn test_random_apply_always() {
        // p=1.0: transforms should always be applied.
        let data = vec![10.0_f64, 20.0, 30.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 3], false).unwrap();
        let ra = RandomApply::<f64>::new(
            vec![Box::new(
                Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap(),
            )],
            1.0,
        )
        .unwrap();
        let out = ra.apply(t).unwrap();
        let d = out.data().unwrap();
        // (10 - 1)/1 = 9
        assert!((d[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_apply_never() {
        // p=0.0: transforms should never be applied.
        let data = vec![10.0_f64, 20.0, 30.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 3], false).unwrap();
        let ra = RandomApply::<f64>::new(
            vec![Box::new(
                Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap(),
            )],
            0.0,
        )
        .unwrap();
        let out = ra.apply(t).unwrap();
        let d = out.data().unwrap();
        assert!((d[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_apply_empty_transforms() {
        // Even with p=1.0, empty transforms should act as identity.
        let data = vec![5.0_f64, 6.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2], false).unwrap();
        let ra = RandomApply::<f64>::new(vec![], 1.0).unwrap();
        let out = ra.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_random_choice_selects_one() {
        // Two transforms: one subtracts 1.0, the other subtracts 100.0.
        // Over many trials, both should be selected at least once.
        let mut saw_small = false;
        let mut saw_large = false;

        for _ in 0..200 {
            let data = vec![500.0_f64];
            let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1], false).unwrap();
            let rc = RandomChoice::<f64>::new(vec![
                Box::new(Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap()),
                Box::new(Normalize::<f64>::new(vec![100.0], vec![1.0]).unwrap()),
            ])
            .unwrap();
            let out = rc.apply(t).unwrap();
            let d = out.data().unwrap();
            if (d[0] - 499.0).abs() < 1e-10 {
                saw_small = true;
            }
            if (d[0] - 400.0).abs() < 1e-10 {
                saw_large = true;
            }
        }

        assert!(saw_small, "RandomChoice never selected first transform");
        assert!(saw_large, "RandomChoice never selected second transform");
    }

    #[test]
    fn test_random_choice_single_transform() {
        let data = vec![10.0_f64];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1], false).unwrap();
        let rc = RandomChoice::<f64>::new(vec![Box::new(
            Normalize::<f64>::new(vec![5.0], vec![1.0]).unwrap(),
        )])
        .unwrap();
        let out = rc.apply(t).unwrap();
        let d = out.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_apply_is_send_sync() {
        fn assert_send_sync<U: Send + Sync>() {}
        assert_send_sync::<RandomApply<f32>>();
        assert_send_sync::<RandomChoice<f32>>();
    }

    #[test]
    fn test_random_choice_weighted_skews_distribution() {
        // Heavily-weighted second transform should dominate the empirical
        // selection counts across many trials.
        let mut saw_small = 0;
        let mut saw_large = 0;
        for _ in 0..1000 {
            let data = vec![500.0_f64];
            let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1], false).unwrap();
            let rc = RandomChoice::<f64>::new(vec![
                Box::new(Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap()),
                Box::new(Normalize::<f64>::new(vec![100.0], vec![1.0]).unwrap()),
            ])
            .unwrap()
            .with_p(vec![0.1, 0.9])
            .unwrap();
            let out = rc.apply(t).unwrap();
            let d = out.data().unwrap();
            if (d[0] - 499.0).abs() < 1e-10 {
                saw_small += 1;
            } else if (d[0] - 400.0).abs() < 1e-10 {
                saw_large += 1;
            }
        }
        // With p=[0.1, 0.9], the second transform should be selected much
        // more often than the first.
        assert!(
            saw_large > saw_small * 3,
            "expected weighted skew (small={saw_small}, large={saw_large})"
        );
    }

    #[test]
    fn test_random_choice_with_p_length_mismatch_errors() {
        let rc = RandomChoice::<f64>::new(vec![Box::new(
            Normalize::<f64>::new(vec![5.0], vec![1.0]).unwrap(),
        )])
        .unwrap()
        .with_p(vec![0.5, 0.5]);
        assert!(rc.is_err());
    }

    #[test]
    fn test_random_choice_with_p_all_zero_errors() {
        let rc = RandomChoice::<f64>::new(vec![
            Box::new(Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap()),
            Box::new(Normalize::<f64>::new(vec![2.0], vec![1.0]).unwrap()),
        ])
        .unwrap()
        .with_p(vec![0.0, 0.0]);
        assert!(rc.is_err());
    }

    #[test]
    fn test_random_choice_with_p_negative_weight_errors() {
        let rc = RandomChoice::<f64>::new(vec![
            Box::new(Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap()),
            Box::new(Normalize::<f64>::new(vec![2.0], vec![1.0]).unwrap()),
        ])
        .unwrap()
        .with_p(vec![-0.1, 0.5]);
        assert!(rc.is_err());
    }
}
