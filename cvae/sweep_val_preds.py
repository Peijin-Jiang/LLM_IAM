"""
sweep_val_preds.py
──────────────────
Regenerate the VAL prediction CSV for EVERY finished sweep cell of one
condition, from each cell's saved best-epoch checkpoint (best_cvae_model.pth).
NO retraining - checkpoint selection already happened during the sweep; this
is one forward pass per cell so that the per-config bootstrap (B=1000) can be
computed for the hyperparameter-selection table.

Sampling protocol matches the sweep's val scoring: one FIXED decoder seed
(CVAE_VAL_EVAL_SEED, default 42) shared by every cell - common random numbers.

Usage (from the project root):
    python3 cvae/sweep_val_preds.py --condition 1
Output per cell dir:  AR6_R10sd_val10_pred_cvae_sweep_seed<seed>.csv
Skips cells whose CSV already exists (delete it to redo).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

import train_cvae as tc
                                  # run as  python3 cvae/sweep_val_preds.py
                                  # (this file sits next to train_cvae.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, choices=[1, 2], default=1)
    ap.add_argument("--seed", type=int, default=tc.VAL_EVAL_SEED)
    args = ap.parse_args()
    mode = args.condition

    root = tc._ROOT / "outputs" / "cvae"
    pat = re.compile(rf"^cond{mode}_lr([0-9.e-]+)_hid(\d+)_lat(\d+)_beta([0-9.]+)_")
    cells = []
    for d in sorted(root.iterdir()):
        m = pat.match(d.name)
        if m and (d / "best_cvae_model.pth").exists():
            cells.append((d, float(m.group(1)), int(m.group(2)),
                          int(m.group(3)), float(m.group(4))))
    assert cells, f"no finished cond{mode} cells under {root}"
    print(f"cond{mode}: {len(cells)} finished cells")

    # ── data prep: identical to train_cvae.main() ────────────────────────────
    fingerprints = tc.read_model_fingerprints()
    desc_dict = tc.load_desc_dict()
    train_cases = tc.read_cases(tc.TRAIN_PATH, fingerprints, desc_dict)
    val_cases = tc.read_cases(tc.VAL_PATH, fingerprints, desc_dict)
    texts = [t for c in train_cases + val_cases
             for t in tc.condition_texts(c, mode)]
    emb = tc.compute_condition_embeddings(texts)   # served from the cache

    X_tr, _ = tc.build_matrices(train_cases)
    X_va, M_va = tc.build_matrices(val_cases)
    mu, sd = tc.fit_standardizer(X_tr)             # TRAIN-only, as in training
    Z_va = tc.transform(X_va, mu, sd)

    from torch.utils.data import DataLoader
    va_ds = tc.cVAEData(val_cases, emb, mode, Z_va, M_va)
    va_loader = DataLoader(va_ds, batch_size=tc.BATCH_SIZE)   # no shuffle
    e0, x0, _ = va_ds[0]
    cond_dim, out_shape = e0.shape[0], x0.shape
    input_dim = int(np.prod(out_shape))

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"val {len(val_cases)} cases | device {device} | decoder seed {args.seed}")

    for d, lr, hidden, latent, beta in cells:
        out_csv = d / f"AR6_R10sd_val10_pred_cvae_sweep_seed{args.seed}.csv"
        if out_csv.exists():
            print(f"  [skip] {d.name} (csv exists)")
            continue
        model = tc.CVAE(input_dim, cond_dim, hidden, latent, out_shape).to(device)
        model.load_state_dict(torch.load(d / "best_cvae_model.pth",
                                         map_location=device))
        torch.manual_seed(args.seed)               # same z draws for every cell
        preds, trues, masks = tc.predict(model, va_loader, device, latent,
                                         mu, sd, X_raw=X_va)
        m = tc.eval_from_table(tc.per_traj_table(preds, trues, masks, val_cases))
        tc.save_pred_csv(preds, val_cases, out_csv)
        print(f"  {d.name}: val MAE {m['mae']:.4f} | sMAPE {m['smape_%']:.2f}% "
              f"-> {out_csv.name}")

    print("done.")


if __name__ == "__main__":
    main()
