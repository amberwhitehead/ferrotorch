//! Phase 2b sentinel (GPU dtype-parity epic, crosslink #1185): `IntTensor<I>`
//! integer COMPUTE on CUDA — elementwise arithmetic, bitwise, and int→int
//! reductions execute on the GPU (real PTX kernel, result stays resident — NO
//! CPU round trip), and match a PyTorch-correct CPU reference bit-for-bit.
//!
//! What this probe asserts, for i32 AND i64, for EVERY Phase-2b op:
//!   1. Inputs are built on CUDA; the op runs; `result.is_cuda()` holds
//!      (GPU-resident — the op launched a kernel, it did not silently fall
//!      back to host).
//!   2. The result values (read back ONCE, here in the value-check only) equal
//!      a CPU reference computed the PyTorch-correct way.
//!   3. A matching CPU-path run (same ops on CPU IntTensors) equals the SAME
//!      reference — proving CPU and GPU agree.
//!
//! Negative-operand cases are included for the sign-sensitive ops:
//!   floor_divide(-7, 2) == -4, remainder(-7, 2) == 1, remainder(7, -2) == -1,
//!   (-8) >> 1 == -4 (arithmetic shift).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::device::Device;
use ferrotorch_core::int_tensor::{IntElement, IntTensor};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for Phase 2b int-ops probe");
    });
}

// ── PyTorch-correct CPU references (the oracle) ─────────────────────────────

fn ref_floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

fn ref_remainder(a: i64, b: i64) -> i64 {
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        r + b
    } else {
        r
    }
}

/// Build a CPU IntTensor, move to CUDA, run `gpu_op`, assert resident, read
/// back, and compare to `expected`. Also run the same op on the CPU tensor and
/// compare. Records pass/fail into the counters.
#[allow(clippy::too_many_arguments)]
fn check_binary<I: IntElement>(
    label: &str,
    a_host: &[i64],
    b_host: &[i64],
    shape: &[usize],
    expected: &[i64],
    gpu_op: impl Fn(&IntTensor<I>, &IntTensor<I>) -> IntTensor<I>,
    pass: &mut usize,
    fail: &mut usize,
) {
    let to_i = |v: &[i64]| -> Vec<I> {
        v.iter()
            .map(|&x| I::try_from_i64(x).expect("operand fits element width"))
            .collect()
    };
    let exp: Vec<I> = to_i(expected);

    // -- CPU path --
    let a_cpu = IntTensor::<I>::from_vec(to_i(a_host), shape.to_vec()).unwrap();
    let b_cpu = IntTensor::<I>::from_vec(to_i(b_host), shape.to_vec()).unwrap();
    let r_cpu = gpu_op(&a_cpu, &b_cpu);
    let cpu_ok = !r_cpu.is_cuda() && r_cpu.data().unwrap() == exp.as_slice();

    // -- GPU path --
    let a_g = a_cpu.to(Device::Cuda(0)).unwrap();
    let b_g = b_cpu.to(Device::Cuda(0)).unwrap();
    let r_g = gpu_op(&a_g, &b_g);
    let resident = r_g.is_cuda();
    let r_back = r_g.to(Device::Cpu).unwrap();
    let gpu_vals_ok = r_back.data().unwrap() == exp.as_slice();

    let ok = cpu_ok && resident && gpu_vals_ok;
    if ok {
        *pass += 1;
        println!(
            "PASS [{:>3}] {label:<28} (gpu resident={resident}, cpu==ref, gpu==ref)",
            I::dtype_name()
        );
    } else {
        *fail += 1;
        println!(
            "FAIL [{:>3}] {label:<28} cpu_ok={cpu_ok} resident={resident} gpu_vals_ok={gpu_vals_ok}\n  \
             expected={exp:?} cpu={:?} gpu={:?}",
            I::dtype_name(),
            r_cpu.data().unwrap(),
            r_back.data().unwrap(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_unary<I: IntElement>(
    label: &str,
    a_host: &[i64],
    shape: &[usize],
    expected: &[i64],
    gpu_op: impl Fn(&IntTensor<I>) -> IntTensor<I>,
    pass: &mut usize,
    fail: &mut usize,
) {
    let to_i = |v: &[i64]| -> Vec<I> {
        v.iter()
            .map(|&x| I::try_from_i64(x).expect("operand fits element width"))
            .collect()
    };
    let exp: Vec<I> = to_i(expected);

    let a_cpu = IntTensor::<I>::from_vec(to_i(a_host), shape.to_vec()).unwrap();
    let r_cpu = gpu_op(&a_cpu);
    let cpu_ok = !r_cpu.is_cuda() && r_cpu.data().unwrap() == exp.as_slice();

    let a_g = a_cpu.to(Device::Cuda(0)).unwrap();
    let r_g = gpu_op(&a_g);
    let resident = r_g.is_cuda();
    let r_back = r_g.to(Device::Cpu).unwrap();
    let gpu_vals_ok = r_back.data().unwrap() == exp.as_slice();

    let ok = cpu_ok && resident && gpu_vals_ok;
    if ok {
        *pass += 1;
        println!(
            "PASS [{:>3}] {label:<28} (gpu resident={resident}, cpu==ref, gpu==ref)",
            I::dtype_name()
        );
    } else {
        *fail += 1;
        println!(
            "FAIL [{:>3}] {label:<28} cpu_ok={cpu_ok} resident={resident} gpu_vals_ok={gpu_vals_ok}\n  \
             expected={exp:?} cpu={:?} gpu={:?}",
            I::dtype_name(),
            r_cpu.data().unwrap(),
            r_back.data().unwrap(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_reduce<I: IntElement>(
    label: &str,
    a_host: &[i64],
    expected: i64,
    gpu_op: impl Fn(&IntTensor<I>) -> IntTensor<I>,
    pass: &mut usize,
    fail: &mut usize,
) {
    let data: Vec<I> = a_host
        .iter()
        .map(|&x| I::try_from_i64(x).expect("operand fits element width"))
        .collect();
    let exp = I::try_from_i64(expected).expect("expected fits element width");

    let a_cpu = IntTensor::<I>::from_vec(data, vec![a_host.len()]).unwrap();
    let r_cpu = gpu_op(&a_cpu);
    let cpu_ok = !r_cpu.is_cuda() && r_cpu.numel() == 1 && r_cpu.data().unwrap()[0] == exp;

    let a_g = a_cpu.to(Device::Cuda(0)).unwrap();
    let r_g = gpu_op(&a_g);
    let resident = r_g.is_cuda();
    let r_back = r_g.to(Device::Cpu).unwrap();
    let gpu_vals_ok = r_back.numel() == 1 && r_back.data().unwrap()[0] == exp;

    let ok = cpu_ok && resident && gpu_vals_ok;
    if ok {
        *pass += 1;
        println!(
            "PASS [{:>3}] {label:<28} (gpu resident={resident}, scalar cpu==ref==gpu)",
            I::dtype_name()
        );
    } else {
        *fail += 1;
        println!(
            "FAIL [{:>3}] {label:<28} cpu_ok={cpu_ok} resident={resident} gpu_vals_ok={gpu_vals_ok} \
             expected={expected}",
            I::dtype_name()
        );
    }
}

/// Exercise every Phase-2b op for one element width.
fn run_for_width<I: IntElement>(pass: &mut usize, fail: &mut usize) {
    // Operands chosen to include negatives and sign-mixing for the
    // sign-sensitive ops. Shape [6] for elementwise.
    let a = [7_i64, -7, 8, -8, 100, -100];
    let b = [2_i64, 2, -2, -2, 7, 7];
    let shape = [6usize];

    // add / sub / mul
    check_binary::<I>(
        "add",
        &a,
        &b,
        &shape,
        &[9, -5, 6, -10, 107, -93],
        |x, y| x.add(y).unwrap(),
        pass,
        fail,
    );
    check_binary::<I>(
        "sub",
        &a,
        &b,
        &shape,
        &[5, -9, 10, -6, 93, -107],
        |x, y| x.sub(y).unwrap(),
        pass,
        fail,
    );
    check_binary::<I>(
        "mul",
        &a,
        &b,
        &shape,
        &[14, -14, -16, 16, 700, -700],
        |x, y| x.mul(y).unwrap(),
        pass,
        fail,
    );

    // floor_divide / remainder — the negative-operand correctness cases.
    let fd: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| ref_floor_div(x, y)).collect();
    check_binary::<I>(
        "floor_div",
        &a,
        &b,
        &shape,
        &fd,
        |x, y| x.floor_div(y).unwrap(),
        pass,
        fail,
    );
    // Spot-check the canonical value from the spec: -7 floor_div 2 == -4.
    assert_eq!(ref_floor_div(-7, 2), -4, "reference floor_div(-7,2)");
    let rem: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| ref_remainder(x, y)).collect();
    check_binary::<I>(
        "remainder",
        &a,
        &b,
        &shape,
        &rem,
        |x, y| x.remainder(y).unwrap(),
        pass,
        fail,
    );
    // Spec spot-checks: remainder(-7,2)==1, remainder(7,-2)==-1.
    assert_eq!(ref_remainder(-7, 2), 1, "reference remainder(-7,2)");
    assert_eq!(ref_remainder(7, -2), -1, "reference remainder(7,-2)");

    // bitwise and / or / xor
    let band: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| x & y).collect();
    check_binary::<I>("bitand", &a, &b, &shape, &band, |x, y| x.bitand(y).unwrap(), pass, fail);
    let bor: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| x | y).collect();
    check_binary::<I>("bitor", &a, &b, &shape, &bor, |x, y| x.bitor(y).unwrap(), pass, fail);
    let bxor: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect();
    check_binary::<I>("bitxor", &a, &b, &shape, &bxor, |x, y| x.bitxor(y).unwrap(), pass, fail);

    // shl / shr (arithmetic). Shift counts in-range; include a negative
    // operand for shr so the sign-extension is exercised: -8 >> 1 == -4.
    let sa = [1_i64, -8, 5, -1, 16, -16];
    let sb = [3_i64, 1, 2, 0, 1, 2];
    let shl_exp: Vec<i64> = sa
        .iter()
        .zip(sb.iter())
        .map(|(&x, &s)| {
            if I::BITS == 32 {
                ((x as i32) << s) as i64
            } else {
                x << s
            }
        })
        .collect();
    check_binary::<I>("shl", &sa, &sb, &shape, &shl_exp, |x, y| x.shl(y).unwrap(), pass, fail);
    let shr_exp: Vec<i64> = sa
        .iter()
        .zip(sb.iter())
        .map(|(&x, &s)| {
            if I::BITS == 32 {
                ((x as i32) >> s) as i64 // arithmetic on signed
            } else {
                x >> s
            }
        })
        .collect();
    check_binary::<I>("shr (arith)", &sa, &sb, &shape, &shr_exp, |x, y| x.shr(y).unwrap(), pass, fail);
    // Spec spot-check: -8 >> 1 == -4 (arithmetic shift on signed). The operands
    // are literals to document the spec value, so eq_op (constant comparison)
    // is expected here.
    #[allow(clippy::eq_op)]
    {
        assert_eq!((-8_i32) >> 1, -4, "reference -8 shr 1");
    }

    // neg / bitnot
    check_unary::<I>(
        "neg",
        &a,
        &shape,
        &[-7, 7, -8, 8, -100, 100],
        |x| x.neg().unwrap(),
        pass,
        fail,
    );
    let bnot: Vec<i64> = a
        .iter()
        .map(|&x| if I::BITS == 32 { (!(x as i32)) as i64 } else { !x })
        .collect();
    check_unary::<I>("bitnot", &a, &shape, &bnot, |x| x.bitnot().unwrap(), pass, fail);

    // reductions: sum / prod / min / max
    let red = [3_i64, -1, 4, -2, 5];
    check_reduce::<I>("sum", &red, 9, |x| x.sum().unwrap(), pass, fail);
    let prod = [2_i64, -3, 4, -1]; // 24
    check_reduce::<I>("prod", &prod, 24, |x| x.prod().unwrap(), pass, fail);
    check_reduce::<I>("min", &red, -2, |x| x.min().unwrap(), pass, fail);
    check_reduce::<I>("max", &red, 5, |x| x.max().unwrap(), pass, fail);
}

#[test]
fn probe_phase2b_int_ops() {
    ensure_cuda_backend();

    let mut pass = 0usize;
    let mut fail = 0usize;

    println!("── i32 ───────────────────────────────────────────────");
    run_for_width::<i32>(&mut pass, &mut fail);
    println!("── i64 ───────────────────────────────────────────────");
    run_for_width::<i64>(&mut pass, &mut fail);

    println!("──────────────────────────────────────────────────────");
    println!("PASS: {pass}, FAIL: {fail}");
    assert_eq!(fail, 0, "Phase 2b int-ops probe had failures");
}
