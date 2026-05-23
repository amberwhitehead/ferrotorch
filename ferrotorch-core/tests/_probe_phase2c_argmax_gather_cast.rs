//! Phase 2c sentinel (GPU dtype-parity epic, crosslink #1185): cross-world
//! integer ops on CUDA — argmax/argmin, index_select/gather (GPU-resident
//! `IntTensor` index), and dtype casts — execute on the GPU (real PTX kernel,
//! result stays resident, NO CPU round trip) and match a PyTorch-correct CPU
//! reference.
//!
//! The headline assertion is the **end-to-end no-round-trip token path**: an
//! embedding table `Tensor<f32>` + token-id `IntTensor<i64>` both on CUDA →
//! index_select rows (GPU) → a float op (GPU) → argmax(dim=-1) → i64 IntTensor
//! on GPU, with EVERY intermediate `.is_cuda()` asserted. This is the empirical
//! proof of the Llama generation loop's GPU-resident sampling path.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::int_tensor::IntTensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for Phase 2c probe");
    });
}

fn record(pass: &mut usize, fail: &mut usize, name: &str, ok: bool) {
    if ok {
        *pass += 1;
        println!("  PASS  {name}");
    } else {
        *fail += 1;
        println!("  FAIL  {name}");
    }
}

#[test]
fn phase2c_argmax_gather_cast() {
    ensure_cuda_backend();
    let mut pass = 0usize;
    let mut fail = 0usize;
    println!("== Phase 2c: argmax/argmin, index_select/gather, cast ==");

    // ── 1. argmax / argmin on f32, global + along-dim, incl. a tie ──────────
    {
        // global argmax/argmin over a flat f32 tensor
        let cpu = from_vec::<f32>(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], &[7]).unwrap();
        let g = cpu.to(Device::Cuda(0)).unwrap();
        let mx = g.argmax(None).unwrap();
        let mn = g.argmin(None).unwrap();
        let ok = mx.is_cuda()
            && mn.is_cuda()
            && mx.to(Device::Cpu).unwrap().data().unwrap() == [5i64]
            && mn.to(Device::Cpu).unwrap().data().unwrap() == [1i64];
        record(&mut pass, &mut fail, "argmax/argmin f32 global (is_cuda + values)", ok);

        // tie: two maxima -> first index
        let tie = from_vec::<f32>(vec![5.0, 1.0, 5.0, 2.0], &[4]).unwrap().to(Device::Cuda(0)).unwrap();
        let tmx = tie.argmax(None).unwrap();
        record(
            &mut pass, &mut fail,
            "argmax f32 tie -> first index",
            tmx.is_cuda() && tmx.to(Device::Cpu).unwrap().data().unwrap() == [0i64],
        );

        // along dim=1 on a [2,3] tensor
        let m = from_vec::<f32>(vec![1.0, 9.0, 2.0, 7.0, 3.0, 4.0], &[2, 3]).unwrap()
            .to(Device::Cuda(0)).unwrap();
        let amx = m.argmax(Some(1)).unwrap();
        let ok_dim = amx.is_cuda()
            && amx.shape() == [2]
            && amx.to(Device::Cpu).unwrap().data().unwrap() == [1i64, 0i64];
        record(&mut pass, &mut fail, "argmax f32 dim=1 (shape [2] + values)", ok_dim);
    }

    // ── argmax/argmin on i32 (value dtype = int) ────────────────────────────
    {
        let cpu = IntTensor::<i32>::from_vec(vec![-3, 7, 7, 2], vec![4]).unwrap();
        let g = cpu.to(Device::Cuda(0)).unwrap();
        let mx = g.argmax(None).unwrap();
        let mn = g.argmin(None).unwrap();
        let ok = mx.is_cuda()
            && mn.is_cuda()
            && mx.to(Device::Cpu).unwrap().data().unwrap() == [1i64] // first 7
            && mn.to(Device::Cpu).unwrap().data().unwrap() == [0i64]; // -3
        record(&mut pass, &mut fail, "argmax/argmin i32 global (is_cuda + values + tie)", ok);
    }

    // ── 2. index_select(dim=0) + gather using a GPU-resident IntTensor ──────
    {
        // index_select dim=0: table [4,2], pick rows [2,0,2]
        let table = from_vec::<f32>(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[4, 2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let idx = IntTensor::<i64>::from_vec(vec![2, 0, 2], vec![3])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let sel = table.index_select(0, &idx).unwrap();
        let ok_sel = sel.is_cuda()
            && sel.shape() == [3, 2]
            && sel.to(Device::Cpu).unwrap().data_vec().unwrap()
                == vec![4.0f32, 5.0, 0.0, 1.0, 4.0, 5.0];
        record(&mut pass, &mut fail, "index_select(dim=0) f32 + i64 idx (is_cuda + values)", ok_sel);

        // gather dim=1 on a [2,3] table with a [2,2] index
        let t2 = from_vec::<f32>(vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0], &[2, 3])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let gidx = IntTensor::<i64>::from_vec(vec![0, 2, 2, 1], vec![2, 2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let g = t2.gather(1, &gidx).unwrap();
        let ok_gather = g.is_cuda()
            && g.shape() == [2, 2]
            && g.to(Device::Cpu).unwrap().data_vec().unwrap()
                == vec![10.0f32, 12.0, 22.0, 21.0];
        record(&mut pass, &mut fail, "gather(dim=1) f32 + i64 idx (is_cuda + values)", ok_gather);
    }

    // ── 3. casts: f32->i32 (truncate), i32->f32, i32->i64, f32->i64 ─────────
    {
        let f = from_vec::<f32>(vec![1.9, -1.9, 2.0, -2.5], &[4])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let i32t = f.to_int::<i32>().unwrap();
        let ok_fi = i32t.is_cuda()
            && i32t.to(Device::Cpu).unwrap().data().unwrap() == [1i32, -1, 2, -2];
        record(&mut pass, &mut fail, "cast f32->i32 truncate (is_cuda + values)", ok_fi);

        let i64t = f.to_int::<i64>().unwrap();
        let ok_fi64 = i64t.is_cuda()
            && i64t.to(Device::Cpu).unwrap().data().unwrap() == [1i64, -1, 2, -2];
        record(&mut pass, &mut fail, "cast f32->i64 truncate (is_cuda + values)", ok_fi64);

        let ii = IntTensor::<i32>::from_vec(vec![-5, 7, 0], vec![3])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let ff = ii.to_float::<f32>().unwrap();
        let ok_if = ff.is_cuda()
            && ff.to(Device::Cpu).unwrap().data_vec().unwrap() == vec![-5.0f32, 7.0, 0.0];
        record(&mut pass, &mut fail, "cast i32->f32 (is_cuda + values)", ok_if);

        let widened = ii.cast::<i64>().unwrap();
        let ok_ii = widened.is_cuda()
            && widened.to(Device::Cpu).unwrap().data().unwrap() == [-5i64, 7, 0];
        record(&mut pass, &mut fail, "cast i32->i64 GPU (is_cuda + values)", ok_ii);
    }

    // ── 4. END-TO-END no-round-trip Llama token path ───────────────────────
    // embedding table f32 (vocab=4, dim=3) + token ids i64 on GPU
    //   -> index_select rows (GPU) -> relu (GPU float op) -> argmax(dim=-1) i64
    // Every intermediate must be is_cuda(); final indices match a CPU reference.
    {
        println!("-- end-to-end no-round-trip token path --");
        let table_host = vec![
            0.1f32, 0.9, 0.2, // row 0
            0.5, 0.4, 0.6, // row 1
            -0.3, 0.8, 0.7, // row 2
            0.2, 0.1, 0.95, // row 3
        ];
        let table = from_vec::<f32>(table_host.clone(), &[4, 3])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let tokens = IntTensor::<i64>::from_vec(vec![3, 0, 2], vec![3])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();

        assert!(table.is_cuda(), "embedding table must be CUDA-resident");
        assert!(tokens.is_cuda(), "token ids must be CUDA-resident");
        println!("  residency: table.is_cuda()={} tokens.is_cuda()={}", table.is_cuda(), tokens.is_cuda());

        // 1) gather embedding rows for the tokens (GPU)
        let embeds = table.index_select(0, &tokens).unwrap();
        assert!(embeds.is_cuda(), "index_select output must stay on CUDA (no round trip)");
        println!("  residency: embeds.is_cuda()={} shape={:?}", embeds.is_cuda(), embeds.shape());

        // 2) a float op (GPU)
        let activated = embeds.relu().unwrap();
        assert!(activated.is_cuda(), "relu output must stay on CUDA (no round trip)");
        println!("  residency: activated.is_cuda()={}", activated.is_cuda());

        // 3) argmax along the last dim -> next-token-style i64 indices (GPU)
        let next = activated.argmax(Some(-1)).unwrap();
        assert!(next.is_cuda(), "argmax output must stay on CUDA (no round trip)");
        println!("  residency: next_indices.is_cuda()={} shape={:?}", next.is_cuda(), next.shape());

        // CPU reference: rows [3,0,2] -> relu -> argmax over dim
        let reference: Vec<i64> = [3usize, 0, 2]
            .iter()
            .map(|&r| {
                let row = &table_host[r * 3..r * 3 + 3];
                let relu_row: Vec<f32> = row.iter().map(|&v| v.max(0.0)).collect();
                let mut best = 0i64;
                for j in 1..3 {
                    if relu_row[j] > relu_row[best as usize] {
                        best = j as i64;
                    }
                }
                best
            })
            .collect();
        let got = next.to(Device::Cpu).unwrap().data().unwrap().to_vec();
        let ok_e2e = next.is_cuda() && next.shape() == [3] && got == reference;
        println!("  e2e indices: got={got:?} reference={reference:?}");
        record(&mut pass, &mut fail, "END-TO-END token path: every intermediate is_cuda + final values", ok_e2e);
    }

    // ── 5. CPU paths agree (parity sanity, no GPU) ──────────────────────────
    {
        let cpu = from_vec::<f32>(vec![1.0, 9.0, 2.0, 7.0, 3.0, 4.0], &[2, 3]).unwrap();
        let amx = cpu.argmax(Some(1)).unwrap();
        record(
            &mut pass, &mut fail,
            "CPU argmax dim=1 matches reference",
            !amx.is_cuda() && amx.to(Device::Cpu).unwrap().data().unwrap() == [1i64, 0i64],
        );
        let ci = from_vec::<f32>(vec![1.9, -1.9], &[2]).unwrap().to_int::<i32>().unwrap();
        record(
            &mut pass, &mut fail,
            "CPU f32->i32 truncate matches reference",
            ci.data().unwrap() == [1i32, -1],
        );
    }

    println!("\n========================================");
    println!("PASS: {pass}, FAIL: {fail}");
    println!("========================================");
    assert_eq!(fail, 0, "Phase 2c probe had {fail} failures");
}
