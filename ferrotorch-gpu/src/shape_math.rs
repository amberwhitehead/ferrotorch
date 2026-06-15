use crate::error::{GpuError, GpuResult};

pub(crate) fn checked_numel(shape: &[usize], op: &'static str) -> GpuResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| GpuError::InvalidState {
            message: format!("{op}: shape product {shape:?} overflows usize"),
        })
}

pub(crate) fn numel(shape: &[usize]) -> usize {
    checked_numel(shape, "numel").expect("numel: shape product overflows usize")
}

pub(crate) fn checked_mul3(a: usize, b: usize, c: usize, op: &'static str) -> GpuResult<usize> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(c))
        .ok_or_else(|| GpuError::InvalidState {
            message: format!("{op}: product {a} * {b} * {c} overflows usize"),
        })
}

pub(crate) fn checked_byte_count(
    count: usize,
    elem_size: usize,
    op: &'static str,
) -> GpuResult<usize> {
    if elem_size == 0 {
        return Err(GpuError::InvalidState {
            message: format!("{op}: element size must be nonzero"),
        });
    }
    count
        .checked_mul(elem_size)
        .ok_or_else(|| GpuError::InvalidState {
            message: format!(
                "{op}: storage size calculation overflowed for {count} elements of {elem_size} bytes"
            ),
        })
}

pub(crate) fn checked_alloc_bytes<T>(count: usize, op: &'static str) -> GpuResult<usize> {
    checked_byte_count(count, std::mem::size_of::<T>(), op)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_numel_rejects_overflow() {
        let err = checked_numel(&[usize::MAX, 2], "shape_math_probe")
            .expect_err("shape product must not wrap");
        assert!(
            format!("{err:?}").contains("overflows usize"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn numel_panics_on_overflow() {
        let result = std::panic::catch_unwind(|| {
            let _ = numel(&[usize::MAX, 2]);
        });
        assert!(result.is_err(), "infallible numel must fail loudly");
    }

    #[test]
    fn checked_mul3_rejects_overflow() {
        let err = checked_mul3(usize::MAX, 2, 1, "mul3_probe")
            .expect_err("three-factor products must not wrap");
        assert!(
            format!("{err:?}").contains("overflows usize"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn checked_byte_count_rejects_overflow() {
        let err = checked_byte_count((usize::MAX / 2) + 1, 2, "byte_probe")
            .expect_err("byte counts must not wrap");
        assert!(
            format!("{err:?}").contains("storage size calculation overflowed"),
            "unexpected error: {err:?}"
        );
    }
}
