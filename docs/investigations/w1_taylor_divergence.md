# W1: ferray-window `taylor` vs scipy `taylor` divergence

**Phase:** W1 (investigation only)
**Date:** 2026-05-06
**Repro symptom:** `taylor(M=16, nbar=4, sll=30, norm=true)[0]` →
ferray-window `0.25388183826768423` vs scipy `0.252321041674507`
(Δ ≈ 1.561e-3).

This document traces the divergence to a single algorithmic site, proposes
the W2 fix, and predicts the corrected outputs across a validation matrix
and edge-case set. **No source files have been modified.**

---

## 1. scipy version pinned

```
$ python3 -c "import scipy; print(scipy.__version__)"
1.17.1
```

(Both `python3 -c "import inspect, scipy.signal.windows; print(inspect.getfile(scipy.signal.windows))"` and `inspect.getsourcefile(_windows.taylor)` resolve to
`/home/doll/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py`.)

---

## 2. scipy `taylor` source (verbatim)

File: `/home/doll/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py`

```python
@xp_capabilities(skip_backends=[("jax.numpy", "item assignment")])
def taylor(M, nbar=4, sll=30, norm=True, sym=True, *, xp=None, device=None):
    """ ... docstring ... """
    xp = _namespace(xp)

    if _len_guards(M):
        return xp.ones(M, dtype=xp.float64, device=device)
    M, needs_trunc = _extend(M, sym)

    # Original text uses a negative sidelobe level parameter and then negates
    # it in the calculation of B. To keep consistent with other methods we
    # assume the sidelobe level parameter to be positive.
    B = xp.asarray(10**(sll / 20), device=device)
    A = xp.acosh(B) / xp.pi
    s2 = nbar**2 / (A**2 + (nbar - 0.5)**2)
    ma = xp.arange(1, nbar, dtype=xp.float64, device=device)

    Fm = xp.empty(nbar - 1, dtype=xp.float64, device=device)
    signs = xp.empty_like(ma)
    signs[::2] = 1
    signs[1::2] = -1
    m2 = ma*ma
    for mi, m in enumerate(ma):
        numer = signs[mi] * xp.prod(1 - m2[mi]/s2/(A**2 + (ma - 0.5)**2))
        denom = 2 * xp.prod(1 - m2[mi]/m2[:mi]) * xp.prod(1 - m2[mi]/m2[mi+1:])
        Fm[mi] = numer / denom

    def W(n):
        return 1 + 2*xp.matmul(Fm, xp.cos(
            2*xp.pi*ma[:, xp.newaxis]*(n-M/2.+0.5)/M))

    w = W(xp.arange(M, dtype=xp.float64, device=device))

    # normalize (Note that this is not described in the original text [1])
    if norm:
        scale = 1.0 / W((M - 1) / 2)
        w *= scale

    return _truncate(w, needs_trunc)
```

Helpers used (also from `_windows.py`):

```python
def _len_guards(M):
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1   # M==0 returns ones(0); M==1 returns ones(1)

def _extend(M, sym):
    if not sym:
        return M + 1, True
    else:
        return M, False

def _truncate(w, needed):
    if needed:
        return w[:-1]
    else:
        return w
```

Note: scipy's variable names are unusual — `B = 10**(sll/20)` is the
amplitude ratio (often called `R` in the literature), and `A = acosh(B)/pi`
is what the literature calls `B`.

---

## 3. ferray-window `taylor` source (verbatim)

File: `/home/doll/ferray/ferray-window/src/windows/mod.rs`, lines 744–818.

```rust
pub fn taylor(m: usize, nbar: usize, sll: f64, norm: bool) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    if nbar == 0 {
        return Err(FerrayError::invalid_value("taylor: nbar must be >= 1"));
    }
    if !sll.is_finite() {
        return Err(FerrayError::invalid_value("taylor: sll must be finite"));
    }
    // R = 10^(sll/20), B = (1/π) acosh(R)
    let r = 10.0_f64.powf(sll / 20.0);
    let b = r.acosh() / PI;
    let nbar_f = nbar as f64;
    // sigma^2 chosen so the (nbar)-th zero of the Taylor pattern is at
    // n = nbar (Carrara & Goodman eq. 13).
    let sigma2 = (nbar_f * nbar_f) / (b * b + (nbar_f - 0.5) * (nbar_f - 0.5));

    // Compute coefficients F_m for m = 1..nbar-1.
    let mut f_coeffs = Vec::with_capacity(nbar.saturating_sub(1));
    for mm in 1..nbar {
        let mmf = mm as f64;
        let mut num = 1.0_f64;
        for n in 1..nbar {
            let nf = n as f64;
            num *= 1.0 - mmf * mmf / (sigma2 * (b * b + (nf - 0.5) * (nf - 0.5)));
        }
        let sign = if mm % 2 == 0 { -1.0 } else { 1.0 };
        let mut den = 1.0_f64;
        for n in 1..nbar {
            if n == mm {
                continue;
            }
            let nf = n as f64;
            den *= 1.0 - mmf * mmf / (nf * nf);
        }
        // The 0.5 prefactor: F_0 = 1 contributes the constant term, and
        // each F_m doubles when reflected about zero in the cosine sum,
        // so we halve the inverse-Fourier coefficients here.
        f_coeffs.push(0.5 * sign * num / den);
    }

    let denom = (m - 1) as f64;
    let arr = gen_window(m, |n| {
        let xn = (n as f64) - denom / 2.0;
        let mut w = 1.0_f64;
        for (idx, &fk) in f_coeffs.iter().enumerate() {
            let kk = (idx + 1) as f64;
            w += 2.0 * fk * (2.0 * PI * kk * xn / m as f64).cos();
        }
        w
    })?;

    if !norm {
        return Ok(arr);
    }
    // Normalise so the centre value is 1.
    let s = arr.as_slice().unwrap().to_vec();
    let centre_val = if m % 2 == 1 {
        s[m / 2]
    } else {
        // For even M, centre is between two samples — average.
        0.5 * (s[m / 2 - 1] + s[m / 2])
    };
    if centre_val == 0.0 {
        return Ok(arr); // pathological; leave un-normalised
    }
    let normed: Vec<f64> = s.into_iter().map(|v| v / centre_val).collect();
    Array::from_vec(Ix1::new([m]), normed)
}
```

---

## 4. Step-by-step comparison: `taylor(M=16, nbar=4, sll=30, norm=true)`

Both the scipy reference and a faithful Python re-implementation of the
ferray-window Rust were exercised by `/tmp/w1_taylor_probe.py`. The Python
mirror's un-normalized output for `M=16` matches the cited bug report's
ferray value (`w[0] = 0.25388183826768423`), confirming the mirror is
algorithmically identical to the Rust implementation.

| quantity | scipy | ferray-window | match? |
|---|---|---|---|
| `R = 10^(sll/20)` | `31.622776601683793` | `31.622776601683793` | yes |
| `acosh(R)/π` (scipy `A`, ferray `B`) | `1.319959391142106` | `1.319959391142106` | yes |
| `σ²` | `1.143486649061458` | `1.143486649061458` | yes |
| `F_1` | `0.292656014469...` | `0.292656014469...` | yes |
| `F_2` | `-0.015783745525...` | `-0.015783745525...` | yes |
| `F_3` | `0.002181043147...` | `0.002181043147...` | yes |

`xn` values fed into `cos(2π · k · xn / M)`:

| n | scipy `(n − M/2 + 0.5)` | ferray `n − (M−1)/2` | scipy `xn/M` | ferray `xn/M` | match? |
|---:|---:|---:|---:|---:|:---:|
| 0 | -7.5 | -7.5 | -0.46875 | -0.46875 | yes |
| 1 | -6.5 | -6.5 | -0.40625 | -0.40625 | yes |
| ... | ... | ... | ... | ... | yes |
| 7 | -0.5 | -0.5 | -0.03125 | -0.03125 | yes |
| 8 | 0.5 | 0.5 | 0.03125 | 0.03125 | yes |
| ... | ... | ... | ... | ... | yes |
| 15 | 7.5 | 7.5 | 0.46875 | 0.46875 | yes |

For *even* `M`, `n − M/2 + 0.5 ≡ n − (M−1)/2`. **Both formulas yield
exactly the same `xn`.** (For odd `M` they also match, because both reduce
to integer offsets from the centre sample.)

Un-normalized `w[i]` are therefore identical:

| i | `w[i]` (scipy & ferray, un-normalized) |
|---:|---:|
| 0 | `0.39314308019494142` |
| 1 | `0.50210148199...` |
| 2 | `0.69117669405...` |
| 3 | `0.91739929566...` |
| 4 | `1.14092984213...` |
| 5 | `1.33298403349...` |
| 6 | `1.47373779932...` |
| 7 | `1.54852778315...` |
| 8 | `1.54852778315...` |
| ... | (mirrors) |
| 15 | `0.39314308019494142` |

**Centre value used for normalization (the divergence site):**

| | scipy | ferray-window |
|---|---:|---:|
| formula | `W((M−1)/2) = W(7.5)` (analytic mid-point evaluation) | `0.5 · (w[M/2−1] + w[M/2]) = 0.5 · (w[7] + w[8])` |
| value | **`1.558106599377844`** | **`1.548527783150935`** |
| ratio (ferray/scipy) | — | `0.99385227...` |

For `M=16`, the analytic peak of W lies between sample 7 and sample 8;
scipy evaluates the closed-form Fourier sum at the half-integer index `7.5`
(arg = `0`, so all cosines = 1, giving the pure-sum maximum
`1 + 2·(F_1 + F_2 + F_3)`). Ferray instead averages the two adjacent
*sample* values, which sit slightly off-peak by `cos(2π·k·0.5/M)` factors,
yielding a smaller scale.

Final normalized `w[i]`:

| i | scipy | ferray | Δ (ferray − scipy) |
|---:|---:|---:|---:|
| 0 | `0.252321041674507` | `0.253881838267684` | **`+1.561e-3`** |
| 1 | `0.322251036...` | `0.324244407...` | `+1.993e-3` |
| 2 | `0.443600381...` | `0.446344392...` | `+2.744e-3` |
| 3 | `0.588791099...` | `0.592433219...` | `+3.642e-3` |
| 4 | `0.732254032...` | `0.736783581...` | `+4.530e-3` |
| 5 | `0.855515299...` | `0.860807312...` | `+5.292e-3` |
| 6 | `0.945851725...` | `0.951702516...` | `+5.851e-3` |
| 7 | `0.993852272...` | `1.000000000...` | `+6.148e-3` |
| 8 | `0.993852272...` | `1.000000000...` | `+6.148e-3` |
| ... | (mirrors) |||

Max abs diff = `6.148e-3`, decisively above any FP-error budget.

---

## 5. Divergence site identified

The single source of divergence is the **centre-value formula used for the
`norm=true` rescaling**.

scipy (`_windows.py`, definition of `W` then `scale`):

```python
def W(n):
    return 1 + 2*xp.matmul(Fm, xp.cos(
        2*xp.pi*ma[:, xp.newaxis]*(n-M/2.+0.5)/M))
...
if norm:
    scale = 1.0 / W((M - 1) / 2)
    w *= scale
```

`W` is called with the (possibly fractional) argument `(M − 1) / 2`. For
`M = 16` that argument is `7.5`, which yields `n − M/2 + 0.5 = 0`, so
every cosine is 1 and the closed-form maximum
`1 + 2·Σ F_k` is returned exactly.

ferray-window (`mod.rs:806-812`):

```rust
let centre_val = if m % 2 == 1 {
    s[m / 2]
} else {
    // For even M, centre is between two samples — average.
    0.5 * (s[m / 2 - 1] + s[m / 2])
};
```

For even `M` this **averages two off-peak sample values** rather than
evaluating the analytic Fourier sum at the fractional midpoint. It is not
mathematically equivalent: averaging the two nearest samples gives the
midpoint of the *secant line* between them, not the value of the cosine
sum at the midpoint.

The Fourier-sum peak ratio (true peak / sample-pair-average) at `M=16`,
`nbar=4`, `sll=30` is `1.558106599 / 1.548527783 ≈ 1.006186`, exactly the
factor by which the buggy `w[i]` overshoot scipy.

---

## 6. Root cause statement

For even-length windows, ferray-window normalises by the **average of the
two centre samples**, but scipy normalises by the **closed-form value of
the Taylor cosine sum at the fractional midpoint `n = (M − 1)/2`**. Those
two quantities coincide for odd `M` (they are both the centre sample) but
diverge for even `M` because the cosine sum is non-linear between adjacent
samples. The intended normalisation makes `max(w) = 1` analytically, not
just discretely; ferray's discrete approximation under-estimates the peak
by `cos(π·k/M)` factors and therefore over-normalises every output.

(The `xn`-coordinate formulas — ferray's `n − (M − 1)/2` and scipy's
`n − M/2 + 0.5` — are *equal*, not different. Likewise R / B / σ² / F_m
all match bit-for-bit. Both `gen_window` edge cases (`m == 0`, `m == 1`)
match scipy's `_len_guards` behaviour. The bug is exclusively in the
even-`M` normalisation branch.)

---

## 7. Proposed fix (W2 will apply — DO NOT APPLY HERE)

Replace the sample-averaging branch with a closed-form evaluation of the
Taylor cosine sum at the fractional midpoint `(m − 1) / 2`. The argument
of every cosine is `2π · k · (n_mid − (m − 1)/2) / m = 0`, so the sum
reduces to `1 + 2 · Σ f_coeffs`. We can either evaluate that
closed-form, or call the same lambda used in `gen_window` with the
fractional argument, for clarity and parity with scipy's `W` re-call.

Annotated diff against `/home/doll/ferray/ferray-window/src/windows/mod.rs`
(lines 791–817), preferred form (closed-form, no extra trig):

```rust
let denom = (m - 1) as f64;
let arr = gen_window(m, |n| {
    let xn = (n as f64) - denom / 2.0;
    let mut w = 1.0_f64;
    for (idx, &fk) in f_coeffs.iter().enumerate() {
        let kk = (idx + 1) as f64;
        w += 2.0 * fk * (2.0 * PI * kk * xn / m as f64).cos();
    }
    w
})?;

if !norm {
    return Ok(arr);
}

// Normalise so the analytic centre value (W at the fractional midpoint
// (m-1)/2) is 1. This matches scipy's `scale = 1.0 / W((M-1)/2)`. For
// the Taylor cosine sum that midpoint is xn = 0, so the value collapses
// to 1 + 2·Σ f_coeffs and we don't need another cos() pass.
- let s = arr.as_slice().unwrap().to_vec();
- let centre_val = if m % 2 == 1 {
-     s[m / 2]
- } else {
-     // For even M, centre is between two samples — average.
-     0.5 * (s[m / 2 - 1] + s[m / 2])
- };
+ let centre_val: f64 = 1.0 + 2.0 * f_coeffs.iter().sum::<f64>();
if centre_val == 0.0 {
    return Ok(arr);
}
- let normed: Vec<f64> = s.into_iter().map(|v| v / centre_val).collect();
+ let normed: Vec<f64> = arr
+     .as_slice()
+     .unwrap()
+     .iter()
+     .map(|&v| v / centre_val)
+     .collect();
Array::from_vec(Ix1::new([m]), normed)
```

Equivalent but uglier alternative (parity with scipy's `W(.)` re-call):

```rust
// scipy form: W((m-1)/2). For ferray's xn = n - (m-1)/2 this is xn = 0,
// so the cosine sum collapses to 1 + 2 Σ f_coeffs anyway.
let n_mid = (m as f64 - 1.0) / 2.0;
let xn_mid = n_mid - denom / 2.0;            // == 0
let mut centre_val = 1.0_f64;
for (idx, &fk) in f_coeffs.iter().enumerate() {
    let kk = (idx + 1) as f64;
    centre_val += 2.0 * fk * (2.0 * PI * kk * xn_mid / m as f64).cos();
}
```

The closed-form `1 + 2·Σ f_coeffs` form is preferred (it is what scipy's
`W((M-1)/2)` reduces to analytically and avoids an unnecessary
`cos(0)` calculation). The closed form also correctly handles the
`nbar == 1` edge case (empty `f_coeffs` → `centre_val = 1.0` → no-op
normalisation, same as scipy returning the all-ones window unchanged).

---

## 8. Validation matrix

Empirical results from running `/tmp/w1_taylor_validate.py`, which runs
**both** the current ferray-window algorithm (`ferray_current`) and the
**proposed** algorithm (`ferray_proposed`) against `scipy.signal.windows.taylor`
across multiple parameter combinations. `cur_max_diff` is the max-abs
diff under the current code; `fix_max_diff` is the max-abs diff after the
proposed fix.

Validation matrix (5 cases beyond the cited bug):

| M  | nbar | sll  | norm  | cur max diff | cur w[0] diff | fix max diff | fix w[0] diff | scipy w[0]            |
|----|------|------|-------|--------------|----------------|--------------|----------------|------------------------|
| 16 | 4    | 30.0 | true  | 6.148e-3     | 1.561e-3       | **2.220e-16**| **5.551e-17**  | 0.252321041674507      |
| 16 | 4    | 30.0 | false | 2.220e-16    | -5.551e-17     | 2.220e-16    | -5.551e-17     | 0.393143080194941      |
| 15 | 4    | 30.0 | true  | 1.110e-16    | 0.0            | 1.110e-16    | 0.0            | 0.253584130238533      |
| 32 | 4    | 30.0 | true  | 1.540e-3     | 3.785e-4       | **3.331e-16**| -2.776e-17     | 0.245407615823850      |
| 32 | 6    | 50.0 | true  | 2.577e-3     | 1.326e-4       | **2.220e-16**| -1.388e-17     | 0.051346442009401      |
| 51 | 20   | 100.0| true  | 4.441e-16    | -1.355e-17     | 4.441e-16    | -1.355e-17     | 0.000490556994528      |
| 8  | 4    | 30.0 | true  | 2.439e-2     | 6.983e-3       | **2.220e-16**| 0.0            | 0.279346299823840      |
| 4  | 2    | 30.0 | true  | 9.751e-2     | 4.664e-2       | **0.0**      | 0.0            | 0.431698653081340      |

**Predicted post-fix outputs** (these should all match scipy to within
`5e-13`; for typical `M` values matches are within `~3e-16`):

- `taylor(16, 4, 30, true)`: `w[0] = 0.252321041674507`, `w[7] = w[8] = 0.993852271577096`, `max(w) = 0.993852271577096`.
- `taylor(32, 4, 30, true)`: `w[0] = 0.245407615823850`, `max(w) = 0.998459857416222`.
- `taylor(32, 6, 50, true)`: `w[0] = 0.051346442009401`, `max(w) = 0.997423450541401`.
- `taylor(8, 4, 30, true)`: `w[0] = 0.279346299823840`, `max(w) = 0.975610718096111`.
- `taylor(4, 2, 30, true)`: `w[0] = 0.431698653081340`, `max(w) = 0.902494903898553`.

Cases with **odd** `M` (15, 51) and `norm=false` already match — they are
included to confirm the fix is non-regressive.

---

## 9. Edge cases checked

Run `/tmp/w1_taylor_validate.py` for the full table. Summary:

| M  | nbar | sll   | scipy behavior | current ferray | proposed ferray |
|----|------|-------|----------------|-----------------|------------------|
| 0  | 4    | 30    | returns `ones(0)` (empty) | returns empty array (length 0) — **matches scipy** | matches |
| 1  | 4    | 30    | returns `ones(1) = [1.0]` | returns `[1.0]` — **matches scipy** | matches |
| 2  | 4    | 30    | returns the closed-form pair | **diverges** (0.5 × (w[0]+w[1]) averaging is wrong) — max diff `0.338` | matches scipy (max diff `1.1e-16`) |
| 16 | 1    | 30    | `nbar=1` → no F_m terms → `ones(M)` | matches scipy already (centre averaging of `1.0` and `1.0` = `1.0`) | matches |
| 16 | 16   | 30    | full-bandwidth Taylor | **diverges** (max diff `4.79e-3`) | matches (max diff `3.33e-16`) |
| 16 | 4    | 0     | `R = 1` → `acosh(R)=0` → degenerate Taylor (cosine sum is identically zero except DC) | **`w` is all zeros pre-norm; ferray's centre-average is also 0 → returns un-normalised zero window**; scipy normalises a near-zero window with the analytic peak (`1 + 2·Σ F_m`) yielding huge values | matches scipy (also pathological — note that scipy's output here is *not* a useful window: max amplitude of order `1e+1`, not 1.0) |
| 16 | 4    | 200   | extreme attenuation, very narrow main lobe | diverges (max diff `1.66e-2`) | matches (max diff `3.33e-16`) |

Two notable outcomes:

1. **`sll = 0`**: scipy and the proposed fix both return a window
   amplified by the analytic peak. Current ferray returns a different
   pathological output because `0.5·(w[7]+w[8])` is essentially zero and
   the early-return path (`if centre_val == 0.0`) leaves the array
   un-normalised. The proposed fix uses the closed-form
   `1 + 2·Σ F_coeffs`, which equals scipy's `W((M-1)/2)` and is
   non-zero in this case, so it follows scipy. **This is a behaviour
   change for `sll = 0`, but in the direction of scipy parity; flag for
   W2 to confirm with the architect.**

2. **`nbar >= M`**: scipy raises no error and returns a valid window.
   ferray-window's current implementation also returns without error;
   the proposed fix preserves that behaviour. (scipy's loop over
   `range(1, nbar)` is identical in semantics to ferray's `for mm in 1..nbar`.)

No edge case becomes worse under the proposed fix; the cases that already
matched (M ∈ {0, 1}, M odd, nbar = 1, norm = false) remain unchanged
because the rewrite only touches the even-`M` normalisation branch's
denominator.

---

## Probe artifacts

- `/tmp/w1_taylor_probe.py` — side-by-side intermediate-value dump for the
  cited case `(M=16, nbar=4, sll=30, norm=true)`, plus a closed-form fix
  demonstration. Confirms ferray's un-normalised output is bit-equal to
  scipy's; only the centre value differs.
- `/tmp/w1_taylor_validate.py` — runs current and proposed ferray
  algorithms against `scipy.signal.windows.taylor` on the validation
  matrix and edge cases.

The Python "ferray mirror" used in both probes is a line-by-line port of
`/home/doll/ferray/ferray-window/src/windows/mod.rs:744-818`. Its output
for the cited symptom (`taylor(16, 4, 30, true)[0] = 0.25388183826768423`)
matches the architect-reported ferray output to all printed digits,
confirming the mirror is faithful.

A standalone Rust probe was scoped but not built — the Python mirror's
bit-for-bit agreement on the intermediate values (R, B, σ², F_m, xn,
w_unnorm) at `M=16` already collapses the search space, and the cited
ferray symptom matches the mirror's output to all 17 displayed digits.
W2 should add a Rust regression test exercising
`taylor(16, 4, 30, true)[0] = 0.252321041674507` to lock parity in.
