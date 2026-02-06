# train.py
from __future__ import annotations
import os
import argparse
import torch
from torch import nn
from torch.optim import Adam
from src.losses import gradients, pde_loss, mono_loss

from src.dataset_csv import list_csvs, make_dataloader_from_files
from src.models import FNet, GNet
from src.losses import total_loss  # assumes signature total_loss(...)
import inspect

def call_fnet(fnet, X):
    """
    Calls fnet in a way that matches its forward() signature.
    If forward(self, X): use fnet(X)
    If forward(self, t, x): split X into t and x (first column as t).
    """
    sig = inspect.signature(fnet.forward)
    # parameters excluding self
    params = [p for p in sig.parameters.values() if p.name != "self"]

    if len(params) == 1:
        return fnet(X)

    if len(params) == 2:
        # Common PINN pattern: forward(t, x)
        t = X[:, :1]     # first feature = "cycle index" (acts like t)
        x = X[:, 1:]     # remaining features
        return fnet(t, x)

    raise TypeError(f"Unsupported FNet.forward signature: {sig}")


def train_one_epoch(
    fnet: nn.Module,
    gnet: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    loss_weights: dict,
):
    fnet.train(); gnet.train()
    total = 0.0
    n = 0

    for X1, Y1, X2, Y2 in loader:
        X1, Y1, X2, Y2 = X1.to(device), Y1.to(device), X2.to(device), Y2.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Your total_loss should combine:
        # - data loss between fnet(X1) and Y1
        # - pde loss using gnet / gradients (depends on your implementation)
        # - monotonic loss
        # and return scalar loss + (optional) dict of components
        from src.losses import pde_loss, mono_loss  # add at top of file once

        u_hat = call_fnet(fnet, X1)

        u_true = Y1
        # Make sure X1 requires grad so we can compute derivatives w.r.t inputs
        X1 = X1.clone().detach().requires_grad_(True)

# split inputs: first column is "cycle index" (t), rest is features (x)
        t = X1[:, :1]
        x = X1[:, 1:]

# forward (use whatever call pattern matches your FNet)
        u_hat = fnet(t, x) if fnet.forward.__code__.co_argcount == 3 else fnet(X1)  # simple check
        u_true = Y1

# compute derivatives of u_hat w.r.t t and x
        u_t, u_x = gradients(u_hat, t, x)  # <-- this matches how your gradients() was designed

        lpde = pde_loss(fnet, gnet, X1, X2)   # adjust args if needed
        lmono = mono_loss(fnet, X1)           # adjust args if needed

        loss, ldata = total_loss(u_hat, u_true, lpde, lmono, alpha=1.0, beta=1.0)



        if isinstance(out, tuple):
            loss = out[0]
        else:
            loss = out

        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu())
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def evaluate(fnet: nn.Module, loader, device: torch.device):
    fnet.eval()
    mse = 0.0
    n = 0
    for X1, Y1, X2, Y2 in loader:
        X1, Y1 = X1.to(device), Y1.to(device)
        pred = call_fnet(fnet, X1)

        mse += float(((pred - Y1) ** 2).mean().detach().cpu())
        n += 1
    return mse / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.csv")
    ap.add_argument("--nominal_capacity", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--normalization", action="store_true")
    ap.add_argument("--norm_method", type=str, default="minmax", choices=["minmax", "zscore"])
    ap.add_argument("--save_dir", type=str, default="results")
    # loss weights (match your losses.py)
    ap.add_argument("--w_data", type=float, default=1.0)
    ap.add_argument("--w_pde", type=float, default=1.0)
    ap.add_argument("--w_mono", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = list_csvs(args.data_dir, pattern=args.pattern)
    print("[INFO] Found files:", files)


    # DUPLICATION MODE (robust): if only one file, use it for both train/test.
    # If multiple files, still okay to keep last one for test for now.
    if len(files) == 1:
        train_files = files
        test_files = files
    else:
        train_files = files[:-1]
        test_files = files[-1:]
    print("[INFO] Train files:", train_files)
    print("[INFO] Test files:", test_files)

    train_loader = make_dataloader_from_files(
        train_files,
        nominal_capacity=args.nominal_capacity,
        batch_size=args.batch_size,
        shuffle=True,
        normalization=args.normalization,
        normalization_method=args.norm_method,
    )
    test_loader = make_dataloader_from_files(
        test_files,
        nominal_capacity=args.nominal_capacity,
        batch_size=args.batch_size,
        shuffle=False,
        normalization=args.normalization,
        normalization_method=args.norm_method,
    )

    # Build models
    # NOTE: input dim = number of feature columns (including "cycle index")
    sample_batch = next(iter(train_loader))
    xdim = sample_batch[0].shape[1]

    fnet = FNet(input_dim=xdim).to(device) if "input_dim" in FNet.__init__.__code__.co_varnames else FNet().to(device)
    gnet = GNet(input_dim=xdim).to(device) if "input_dim" in GNet.__init__.__code__.co_varnames else GNet().to(device)

    optimizer = Adam(list(fnet.parameters()) + list(gnet.parameters()), lr=args.lr)

    loss_weights = dict(w_data=args.w_data, w_pde=args.w_pde, w_mono=args.w_mono)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(fnet, gnet, train_loader, optimizer, device, loss_weights)
        te = evaluate(fnet, test_loader, device)

        print(f"Epoch {ep:03d} | train_loss={tr:.6f} | test_mse={te:.6f}")

        if te < best:
            best = te
            ckpt = {
                "epoch": ep,
                "fnet": fnet.state_dict(),
                "gnet": gnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "xdim": xdim,
                "best_test_mse": best,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))

    print(f"Done. Best test MSE: {best:.6f}")
    print(f"Saved: {os.path.join(args.save_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
