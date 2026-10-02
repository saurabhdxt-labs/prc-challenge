"""The ADS-B stack's MLP half, trained and served in its own process (BC-5: torch and LightGBM must never share a
process — on macOS their two OpenMP runtimes deadlock in torch indexing).

    python3.11 scripts/adsb_nn_worker.py IN.npz OUT.npz

IN.npz: A (fit inputs), ya (fit targets, already scaled), B / yb (early-stopping set), T (rows to predict), seed, and
hyper-parameters hidden, dropout, lr, batch, epochs, patience, grad_spike. OUT.npz: pred (model output for T, target
units) and meta. A runtime-guard trip (non-finite training or validation loss, non-finite gradient, gradient spike)
writes `error` and exits 3.
This file imports torch and numpy only; it must never import lightgbm.
"""
from __future__ import annotations

import sys

import numpy as np


def _net(n_in, hidden, dropout):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                               torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                               torch.nn.Linear(hidden, 1))


def train_predict(A, ya, B, yb, T, seed, hidden, dropout, lr, batch, epochs, patience, grad_spike, threads=4,
                  save_path=None):
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if not (np.isfinite(ya).all() and np.isfinite(yb).all()):
        raise FloatingPointError("non-finite loss: the residual target holds a non-finite value")
    net = _net(A.shape[1], hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    At, yat = torch.from_numpy(A), torch.from_numpy(ya.astype(np.float32)).unsqueeze(1)
    Bt, ybt = torch.from_numpy(B), torch.from_numpy(yb.astype(np.float32)).unsqueeze(1)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, ema, step, ep_done = np.inf, None, 0, None, 0, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(At), generator=gen)
        for i in range(0, len(At), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(At[b]), yat[b])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters() if p.grad is not None)))
            if not np.isfinite(gn):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            if ema is not None and step >= 100 and gn > grad_spike * ema:
                raise FloatingPointError(f"gradient spike at step {step}: {gn:.3g} > {grad_spike} x EMA {ema:.3g}")
            ema = gn if ema is None else 0.99 * ema + 0.01 * gn
            opt.step()
            step += 1
        net.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(net(Bt), ybt))
        if not np.isfinite(v):
            raise FloatingPointError(f"non-finite validation loss at epoch {ep}")
        ep_done = ep + 1
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    if save_path:
        torch.save(best_state, save_path)
    with torch.no_grad():
        pred = net(torch.from_numpy(T)).squeeze(1).numpy().astype(np.float64)
    return pred, {"epochs": ep_done, "steps": step, "best_val_mse": best}


def predict_saved(paths, Ts, hidden, dropout, threads=4) -> np.ndarray:
    """Mean prediction of saved state_dicts; model i is applied to its own input matrix Ts[i]."""
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    preds = []
    for path, T in zip(paths, Ts):
        net = _net(T.shape[1], hidden, dropout)
        net.load_state_dict(torch.load(path, weights_only=True))
        net.eval()
        with torch.no_grad():
            preds.append(net(torch.from_numpy(T)).squeeze(1).numpy().astype(np.float64))
    return np.mean(preds, axis=0)


def main(argv) -> int:
    if len(argv) == 3 and argv[0] == "predict":
        z = np.load(argv[1], allow_pickle=False)
        n = int(z["n_models"])
        pred = predict_saved([str(z[f"path_{i}"]) for i in range(n)], [z[f"T_{i}"] for i in range(n)],
                             int(z["hidden"]), float(z["dropout"]))
        np.savez(argv[2], pred=pred)
        return 0
    if len(argv) != 2:
        print("usage: adsb_nn_worker.py IN.npz OUT.npz | predict IN.npz OUT.npz", file=sys.stderr)
        return 2
    z = np.load(argv[0])
    kw = {k: (int(z[k]) if k in ("seed", "hidden", "batch", "epochs", "patience") else float(z[k]))
          for k in ("seed", "hidden", "dropout", "lr", "batch", "epochs", "patience", "grad_spike")}
    try:
        pred, meta = train_predict(z["A"], z["ya"], z["B"], z["yb"], z["T"], **kw,
                                   save_path=str(z["save_path"]) if "save_path" in z.files else None)
    except FloatingPointError as e:
        np.savez(argv[1], error=np.array(str(e)))
        return 3
    np.savez(argv[1], pred=pred, epochs=meta["epochs"], steps=meta["steps"], best_val_mse=meta["best_val_mse"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
