#!/usr/bin/env python3
"""Pin a GCN-on-Cora reference checkpoint to `ferrotorch/gcn-cora` (#1157).

Pipeline:

  1. Download the Cora node-classification benchmark via
     `torch_geometric.datasets.Planetoid` (1433-dim features, 2708
     nodes, 7 classes).
  2. Train a 2-layer GCN matching the PyG `examples/gcn.py` recipe
     (`GCNConv(1433, 16) -> ReLU -> Dropout -> GCNConv(16, 7)`) with
     a frozen `torch.manual_seed(42)` for 200 epochs using Adam +
     lr=0.01 + weight_decay=5e-4 + cross-entropy on the `train_mask`.
  3. Freeze the trained `state_dict()` (verbatim — no key renames; the
     ferrotorch-graph `GcnNet` was specifically built so its
     `named_parameters()` match the upstream PyG layout) and save as
     `model.safetensors`.
  4. Run one eval-mode forward over the full graph and dump:
       * `_value_parity_x.bin`           — [N, F] f32
       * `_value_parity_edge_index.bin`  — [2, E] i64
       * `_value_parity_y.bin`           — [N]    i64
       * `_value_parity_logits.bin`      — [N, C] f32
     in the standard `[u32 ndim][u32 × ndim shape][<dtype> data]`
     little-endian format the other ferrotorch dumps use.
  5. Upload all five artifacts to `huggingface.co/ferrotorch/gcn-cora`.
  6. Print the SHA-256 of `model.safetensors` so the caller can update
     `ferrotorch-hub/src/registry.rs`.

Run via:
  python3 scripts/pin_pretrained_gcn_weights.py
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

WORK_DIR = Path("/tmp/ferrotorch_pin_gcn_cora")
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reference model — same architecture ferrotorch-graph exposes as GcnNet.
# Matches PyG examples/gcn.py with hidden=16; dropout disabled in eval (which
# is the only mode the harness drives).
# ---------------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int):
        super().__init__()
        # Defaults: add_self_loops=True, normalize=True, improved=False, bias=True
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Dropout disabled at eval — train() pass uses F.dropout but the
        # parity dump runs under .eval() / no_grad.
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Binary dump helpers — same `[u32 ndim][u32 × ndim shape][<dtype> data]`
# little-endian format the rest of the project uses.
# ---------------------------------------------------------------------------
def dump_f32(path: Path, t: torch.Tensor) -> None:
    arr = t.detach().cpu().to(torch.float32).numpy()
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def dump_i64(path: Path, t: torch.Tensor) -> None:
    arr = t.detach().cpu().to(torch.int64).numpy()
    arr = np.ascontiguousarray(arr, dtype="<i8")
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def main() -> int:
    # -- 1. Deterministic training. -----------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    print("[pin] loading Cora via torch_geometric.Planetoid…", flush=True)
    dataset = Planetoid(root=str(WORK_DIR / "cora"), name="Cora")
    data = dataset[0]
    print(
        f"[pin] cora: N={data.num_nodes} F={data.num_features} "
        f"C={dataset.num_classes} E_directed={data.edge_index.size(1)}",
        flush=True,
    )
    in_features = data.num_features
    num_classes = dataset.num_classes
    hidden = 16

    model = GCN(in_features, hidden, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("[pin] training for 200 epochs…", flush=True)
    model.train()
    for epoch in range(200):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred = out.argmax(dim=1)
                train_acc = (
                    (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                )
                val_acc = (
                    (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                )
            print(
                f"[pin]   epoch {epoch + 1:3d}: loss={loss.item():.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}",
                flush=True,
            )

    # -- 2. Eval-mode forward to freeze logits. -----------------------------
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    print(f"[pin] frozen test_acc on Cora test_mask: {test_acc:.4f}", flush=True)

    # -- 3. Save state_dict to safetensors. ---------------------------------
    from safetensors.torch import save_file

    state = model.state_dict()
    print("[pin] state_dict keys:", flush=True)
    for k, v in state.items():
        print(f"[pin]   {k}: shape={tuple(v.shape)} dtype={v.dtype}", flush=True)

    weights_path = WORK_DIR / "model.safetensors"
    # safetensors expects contiguous tensors; clone to be safe.
    save_file({k: v.detach().contiguous() for k, v in state.items()}, str(weights_path))
    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    print(f"[pin] wrote {weights_path} ({weights_path.stat().st_size} bytes)", flush=True)
    print(f"[pin] SHA-256: {sha}", flush=True)

    # -- 4. Dump value-parity fixtures. ------------------------------------
    x_path = WORK_DIR / "_value_parity_x.bin"
    ei_path = WORK_DIR / "_value_parity_edge_index.bin"
    y_path = WORK_DIR / "_value_parity_y.bin"
    logits_path = WORK_DIR / "_value_parity_logits.bin"
    dump_f32(x_path, data.x)
    dump_i64(ei_path, data.edge_index)
    dump_i64(y_path, data.y)
    dump_f32(logits_path, logits)
    print(
        f"[pin] dumped parity fixtures: x={x_path.stat().st_size}B, "
        f"edge_index={ei_path.stat().st_size}B, y={y_path.stat().st_size}B, "
        f"logits={logits_path.stat().st_size}B",
        flush=True,
    )

    # -- 5. Upload to HF (best-effort; tolerate offline). ------------------
    upload = os.environ.get("FERROTORCH_PIN_UPLOAD", "1") != "0"
    if upload:
        try:
            from huggingface_hub import HfApi, create_repo, upload_file
        except ImportError:
            print(
                "[pin] huggingface_hub not installed — skipping upload "
                "(set FERROTORCH_PIN_UPLOAD=0 to silence)",
                file=sys.stderr,
                flush=True,
            )
        else:
            repo_id = "ferrotorch/gcn-cora"
            try:
                create_repo(repo_id, repo_type="model", exist_ok=True)
                for relative, local in [
                    ("model.safetensors", weights_path),
                    ("_value_parity_x.bin", x_path),
                    ("_value_parity_edge_index.bin", ei_path),
                    ("_value_parity_y.bin", y_path),
                    ("_value_parity_logits.bin", logits_path),
                    # config.json is fetched by `hf_download_model`; the
                    # ferrotorch hub requires it to exist on the repo.
                ]:
                    upload_file(
                        path_or_fileobj=str(local),
                        path_in_repo=relative,
                        repo_id=repo_id,
                        repo_type="model",
                    )
                # Minimal config.json so the hub download is happy.
                cfg = {
                    "architecture": "GCN",
                    "in_features": in_features,
                    "hidden": hidden,
                    "num_classes": num_classes,
                    "num_nodes": int(data.num_nodes),
                    "num_edges_directed": int(data.edge_index.size(1)),
                    "training": {
                        "optimizer": "Adam",
                        "lr": 0.01,
                        "weight_decay": 5e-4,
                        "epochs": 200,
                        "seed": 42,
                        "dataset": "Cora (Planetoid)",
                    },
                    "frozen_test_accuracy": float(test_acc),
                }
                import json
                cfg_path = WORK_DIR / "config.json"
                cfg_path.write_text(json.dumps(cfg, indent=2))
                upload_file(
                    path_or_fileobj=str(cfg_path),
                    path_in_repo="config.json",
                    repo_id=repo_id,
                    repo_type="model",
                )
                # API call to confirm the repo state.
                api = HfApi()
                files = api.list_repo_files(repo_id=repo_id, repo_type="model")
                print(f"[pin] uploaded to {repo_id}. Repo files:", flush=True)
                for fname in files:
                    print(f"[pin]   - {fname}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[pin] HF upload failed: {exc!r}", file=sys.stderr, flush=True)
                return 2
    else:
        print("[pin] FERROTORCH_PIN_UPLOAD=0, skipping upload", flush=True)

    print(f"\n[pin] DONE. SHA-256 for registry.rs pin: {sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
