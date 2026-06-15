//! Internal port of PyTorch CPU `topk` selection order.
//!
//! PyTorch CPU `topk` is not a stable sort. `TopKImpl.h` builds `(value, index)`
//! pairs and compares only values, then uses libstdc++ `partial_sort` when
//! `k * 64 <= dim_size` and `nth_element` otherwise. Tied indices are therefore
//! an implementation-detail of libstdc++'s heap/select/sort algorithms. Public
//! APIs that expose top-k indices, and pruning masks derived from those indices,
//! need that concrete order for PyTorch parity.

use crate::dtype::Float;

type Pair<T> = (T, usize);

#[inline]
fn topk_lt<T: Float>(x: &Pair<T>, y: &Pair<T>, largest: bool) -> bool {
    if largest {
        (x.0.is_nan() && !y.0.is_nan()) || x.0 > y.0
    } else {
        (!x.0.is_nan() && y.0.is_nan()) || x.0 < y.0
    }
}

fn sift_up<T: Float>(
    v: &mut [Pair<T>],
    mut hole: usize,
    top: usize,
    value: Pair<T>,
    largest: bool,
) {
    while hole > top {
        let parent = (hole - 1) / 2;
        if !topk_lt(&v[parent], &value, largest) {
            break;
        }
        v[hole] = v[parent];
        hole = parent;
    }
    v[hole] = value;
}

fn adjust_heap<T: Float>(
    v: &mut [Pair<T>],
    mut hole: usize,
    len: usize,
    value: Pair<T>,
    largest: bool,
) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if topk_lt(&v[second], &v[second - 1], largest) {
            second -= 1;
        }
        v[hole] = v[second];
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1];
        hole = second - 1;
    }
    sift_up(v, hole, top, value, largest);
}

fn make_heap<T: Float>(v: &mut [Pair<T>], len: usize, largest: bool) {
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[parent];
        adjust_heap(v, parent, len, value, largest);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn heap_select<T: Float>(v: &mut [Pair<T>], middle: usize, largest: bool) {
    make_heap(v, middle, largest);
    for i in middle..v.len() {
        if topk_lt(&v[i], &v[0], largest) {
            let value = v[i];
            v[i] = v[0];
            adjust_heap(&mut v[..middle], 0, middle, value, largest);
        }
    }
}

fn pop_heap<T: Float>(v: &mut [Pair<T>], len: usize, largest: bool) {
    let value = v[len - 1];
    v[len - 1] = v[0];
    adjust_heap(&mut v[..len - 1], 0, len - 1, value, largest);
}

fn sort_heap<T: Float>(v: &mut [Pair<T>], largest: bool) {
    let mut len = v.len();
    while len > 1 {
        pop_heap(v, len, largest);
        len -= 1;
    }
}

fn partial_sort<T: Float>(v: &mut [Pair<T>], middle: usize, largest: bool) {
    heap_select(v, middle, largest);
    sort_heap(&mut v[..middle], largest);
}

fn insertion_sort<T: Float>(v: &mut [Pair<T>], largest: bool) {
    for i in 1..v.len() {
        let value = v[i];
        if topk_lt(&value, &v[0], largest) {
            for j in (1..=i).rev() {
                v[j] = v[j - 1];
            }
            v[0] = value;
        } else {
            let mut j = i;
            while topk_lt(&value, &v[j - 1], largest) {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = value;
        }
    }
}

fn unguarded_insertion_sort<T: Float>(v: &mut [Pair<T>], first: usize, largest: bool) {
    for i in first..v.len() {
        let value = v[i];
        let mut j = i;
        while topk_lt(&value, &v[j - 1], largest) {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = value;
    }
}

fn final_insertion_sort<T: Float>(v: &mut [Pair<T>], largest: bool) {
    const THRESHOLD: usize = 16;
    if v.len() > THRESHOLD {
        insertion_sort(&mut v[..THRESHOLD], largest);
        unguarded_insertion_sort(v, THRESHOLD, largest);
    } else {
        insertion_sort(v, largest);
    }
}

fn move_median_to_first<T: Float>(
    v: &mut [Pair<T>],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    largest: bool,
) {
    if topk_lt(&v[a], &v[b], largest) {
        if topk_lt(&v[b], &v[c], largest) {
            v.swap(result, b);
        } else if topk_lt(&v[a], &v[c], largest) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if topk_lt(&v[a], &v[c], largest) {
        v.swap(result, a);
    } else if topk_lt(&v[b], &v[c], largest) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

fn unguarded_partition<T: Float>(
    v: &mut [Pair<T>],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    largest: bool,
) -> usize {
    loop {
        while topk_lt(&v[first], &v[pivot], largest) {
            first += 1;
        }
        last -= 1;
        while topk_lt(&v[pivot], &v[last], largest) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn unguarded_partition_pivot<T: Float>(v: &mut [Pair<T>], largest: bool) -> usize {
    let len = v.len();
    let mid = len / 2;
    move_median_to_first(v, 0, 1, mid, len - 1, largest);
    unguarded_partition(v, 1, len, 0, largest)
}

fn floor_log2(n: usize) -> usize {
    debug_assert!(n > 0);
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

fn nth_element<T: Float>(v: &mut [Pair<T>], nth: usize, largest: bool) {
    let len = v.len();
    if len == 0 || nth == len {
        return;
    }
    let mut depth_limit = 2 * floor_log2(len);
    let mut first = 0usize;
    let mut last = len;
    while last - first > 3 {
        if depth_limit == 0 {
            heap_select(&mut v[first..last], nth + 1 - first, largest);
            v.swap(first, nth);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(&mut v[first..last], largest) + first;
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    insertion_sort(&mut v[first..last], largest);
}

fn introsort_loop<T: Float>(v: &mut [Pair<T>], mut depth_limit: usize, largest: bool) {
    const THRESHOLD: usize = 16;
    let mut last = v.len();
    while last > THRESHOLD {
        if depth_limit == 0 {
            partial_sort(&mut v[..last], last, largest);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(&mut v[..last], largest);
        introsort_loop(&mut v[cut..last], depth_limit, largest);
        last = cut;
    }
}

fn sort_libstdcpp<T: Float>(v: &mut [Pair<T>], largest: bool) {
    if v.len() > 1 {
        introsort_loop(v, 2 * floor_log2(v.len()), largest);
        final_insertion_sort(v, largest);
    }
}

/// Return PyTorch CPU `topk` `(value, index)` pairs for one last-dimension row.
///
/// This mirrors `aten/src/ATen/native/TopKImpl.h:44-93` for the value/index
/// order. `sorted=false` is included for internal consumers that only need the
/// selected set; the `partial_sort` branch still sorts because upstream does.
pub(crate) fn torch_cpu_topk_pairs<T: Float>(
    values: &[T],
    k: usize,
    largest: bool,
    sorted: bool,
) -> Vec<Pair<T>> {
    debug_assert!(k <= values.len());
    if k == 0 {
        return Vec::new();
    }
    let mut queue: Vec<Pair<T>> = values.iter().copied().zip(0..).collect();
    if (k as u128) * 64 <= values.len() as u128 {
        partial_sort(&mut queue, k, largest);
    } else {
        nth_element(&mut queue, k - 1, largest);
        if sorted {
            // PyTorch sorts `[begin, begin + k - 1)`, leaving the kth element
            // in the selected boundary slot.
            sort_libstdcpp(&mut queue[..k - 1], largest);
        }
    }
    queue.truncate(k);
    queue
}

pub(crate) fn torch_cpu_topk_indices<T: Float>(
    values: &[T],
    k: usize,
    largest: bool,
    sorted: bool,
) -> Vec<usize> {
    torch_cpu_topk_pairs(values, k, largest, sorted)
        .into_iter()
        .map(|(_, idx)| idx)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{torch_cpu_topk_indices, torch_cpu_topk_pairs};

    #[test]
    fn equal_ties_match_torch_nth_element_oracles() {
        assert_eq!(
            torch_cpu_topk_indices(&[1.0_f32, 1.0, 1.0, 1.0], 1, false, true),
            vec![2]
        );
        assert_eq!(
            torch_cpu_topk_indices(&[1.0_f32, 1.0, 1.0, 1.0], 2, false, true),
            vec![2, 3]
        );
        assert_eq!(
            torch_cpu_topk_indices(&vec![1.0_f32; 100], 2, false, true),
            vec![67, 66]
        );
    }

    #[test]
    fn values_and_indices_match_torch_largest_tie_oracle() {
        let got = torch_cpu_topk_pairs(&[4.0_f32, 3.0, 3.0, 4.0], 2, true, true);
        assert_eq!(got, vec![(4.0, 3), (4.0, 0)]);
    }
}
