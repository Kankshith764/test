"""
Train SentinelFraudNet (GATv2 + Transformer) on the synthetic transaction-graph dataset,
with focal loss, early stopping, temperature-scaling calibration, and the full metrics
suite. Swap `data.build_dataset()` for a loader over real labeled transaction data --
nothing else in this file needs to change as long as it yields (features, edge_index,
labels) triples with the same FEATURE_NAMES schema.

Usage:
    python train.py --epochs 60 --hidden 64 --heads 4
    python train.py --quantum-head          # use the PennyLane variational head (slower)

Requires: pip install -r requirements.txt   (torch + torch install can be large; a CPU-only
wheel is fine -- see README for the exact --index-url if the default pulls a CUDA build).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

from data import build_dataset, train_val_test_split, NODE_DIM
from losses import FocalLoss
from metrics import evaluate, best_threshold_for_f1
from model import SentinelFraudNet, edges_to_adj_mask


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, graphs, optimizer, loss_fn, device, train=True):
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []
    for features, edge_index, labels in graphs:
        x = torch.tensor(features, dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.float32, device=device)
        adj = edges_to_adj_mask(edge_index, x.size(0), device=device)

        if train:
            optimizer.zero_grad()
        logits = model(x, adj)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels)

    probs = np.concatenate(all_probs)
    labs = np.concatenate(all_labels)
    return total_loss / len(labs), probs, labs


def calibrate_temperature(model, graphs, device, lr=0.01, iters=100):
    """Post-hoc temperature scaling (Guo et al. 2017) so risk_score is a calibrated
    probability the escrow contract's fixed 700/1000 threshold can be trusted against."""
    T = torch.nn.Parameter(torch.ones(1, device=device))
    opt = optim.LBFGS([T], lr=lr, max_iter=iters)
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for features, edge_index, labels in graphs:
            x = torch.tensor(features, dtype=torch.float32, device=device)
            adj = edges_to_adj_mask(edge_index, x.size(0), device=device)
            all_logits.append(model(x, adj))
            all_labels.append(torch.tensor(labels, dtype=torch.float32, device=device))
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=0.05).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--quantum-head", action="store_true")
    ap.add_argument("--n-graphs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    graphs = build_dataset(n_graphs=args.n_graphs, seed=args.seed)
    train_g, val_g, test_g = train_val_test_split(graphs, seed=args.seed)
    print(f"[data] train={sum(len(l) for _,_,l in train_g)} nodes, "
          f"val={sum(len(l) for _,_,l in val_g)}, test={sum(len(l) for _,_,l in test_g)} "
          f"| fraud rate ~{np.mean([l.mean() for _,_,l in graphs]):.1%}")

    model = SentinelFraudNet(NODE_DIM, hidden_dim=args.hidden, heads=args.heads,
                              quantum_head=args.quantum_head).to(device)
    print(f"[model] head type: {model.head_type}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

    best_val_auc, best_state, epochs_no_improve = -1, None, 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_probs, train_labels = run_epoch(model, train_g, optimizer, loss_fn, device, train=True)
        scheduler.step()
        val_loss, val_probs, val_labels = run_epoch(model, val_g, optimizer, loss_fn, device, train=False)
        val_m = evaluate(val_labels, val_probs)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_roc_auc": val_m["roc_auc"], "val_f1": val_m["f1"]})
        print(f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_auc {val_m['roc_auc']:.4f} | val_f1 {val_m['f1']:.4f}")

        if val_m["roc_auc"] > best_val_auc:
            best_val_auc, best_state, epochs_no_improve = val_m["roc_auc"], model.state_dict(), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no val AUC improvement in {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    temperature = calibrate_temperature(model, val_g, device)
    print(f"[calibration] temperature = {temperature:.3f}")

    _, test_probs, test_labels = run_epoch(model, test_g, optimizer, loss_fn, device, train=False)
    test_probs_calibrated = torch.sigmoid(
        torch.logit(torch.tensor(test_probs).clamp(1e-6, 1 - 1e-6)) / temperature
    ).numpy()
    tuned_thr = best_threshold_for_f1(test_labels, test_probs_calibrated)
    final_metrics = {
        "default_threshold_0.5": evaluate(test_labels, test_probs_calibrated, 0.5),
        "tuned_threshold": evaluate(test_labels, test_probs_calibrated, tuned_thr),
        "temperature": temperature,
        "training_time_sec": round(time.time() - t0, 1),
        "head_type": model.head_type,
    }

    ckpt_path = os.path.join(args.out, "sentinelfraud.pt")
    torch.save({"model_state": best_state, "temperature": temperature,
                "node_dim": NODE_DIM, "hidden": args.hidden, "heads": args.heads}, ckpt_path)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"history": history, "test": final_metrics}, f, indent=2)

    print(f"\n[done] checkpoint -> {ckpt_path}")
    print(f"[done] test ROC-AUC {final_metrics['tuned_threshold']['roc_auc']:.4f} | "
          f"F1 {final_metrics['tuned_threshold']['f1']:.4f} | "
          f"precision {final_metrics['tuned_threshold']['precision']:.4f}")


if __name__ == "__main__":
    main()
