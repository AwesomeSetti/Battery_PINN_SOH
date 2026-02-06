import torch
import torch.nn.functional as F

def data_loss(u_hat, u_true, reduction="mean"):
    """
    L_data = sum ||u_true - u_hat||^2   (usually MSE)

    u_hat: (B,1) predicted SOH
    u_true: (B,1) ground-truth SOH
    """
    return F.mse_loss(u_hat, u_true, reduction=reduction)

def gradients(u_hat, t, x):
    """
    Compute u_t = ∂u/∂t and u_x = ∂u/∂x using autograd.
    u_hat: (B,1)
    t: (B,1) with requires_grad=True
    x: (B,16) with requires_grad=True
    """
    ones = torch.ones_like(u_hat)

    u_t = torch.autograd.grad(
        outputs=u_hat, inputs=t,
        grad_outputs=ones,
        create_graph=True, retain_graph=True
    )[0]  # (B,1)

    u_x = torch.autograd.grad(
        outputs=u_hat, inputs=x,
        grad_outputs=ones,
        create_graph=True, retain_graph=True
    )[0]  # (B,16)

    return u_t, u_x

def pde_loss(gnet, t, x, u_hat, u_t, u_x, reduction="mean"):
    """
    L_PDE = || H ||^2 where H = u_t - G(t, x, u, u_t, u_x)

    gnet: GNet
    t: (B,1)
    x: (B,16)
    u_hat: (B,1)
    u_t: (B,1)
    u_x: (B,16)
    """
    z = torch.cat([t, x, u_hat, u_t, u_x], dim=1)  # (B,35)
    g_hat = gnet(z)                                 # (B,1)
    H = u_t - g_hat                                 # (B,1)
    return (H**2).mean() if reduction == "mean" else (H**2).sum()

def mono_loss(u_hat_seq, reduction="mean"):
    """
    Monotonicity loss: penalize increases in predicted SOH along time.
    u_hat_seq: (T,) or (T,1) or (B,T) or (B,T,1)
      - assumes last dimension (if present) is SOH
      - assumes time is along dimension 0 for 1D/2D, or dimension 1 for batched (B,T,...)

    We compute ReLU(u_{k+1} - u_k) and sum/mean.
    """
    # Squeeze last dim if it's 1
    if u_hat_seq.dim() >= 2 and u_hat_seq.shape[-1] == 1:
        u_hat_seq = u_hat_seq.squeeze(-1)

    # Determine time dimension
    if u_hat_seq.dim() == 1:          # (T,)
        diffs = u_hat_seq[1:] - u_hat_seq[:-1]
    elif u_hat_seq.dim() == 2:        # (T, D) or (B, T)
        # We assume (B,T) is more common; if it's (T,D) you'll handle later explicitly.
        diffs = u_hat_seq[:, 1:] - u_hat_seq[:, :-1]
    else:                              # (B,T, ...)
        diffs = u_hat_seq[:, 1:] - u_hat_seq[:, :-1]

    penalties = torch.relu(diffs)
    if reduction == "mean":
        return penalties.mean()
    elif reduction == "sum":
        return penalties.sum()
    else:
        return penalties

def total_loss(
    u_hat, u_true,
    lpde,
    lmono,
    alpha=1.0,
    beta=1.0
):
    """
    L = L_data + alpha * L_PDE + beta * L_mono
    """
    ldata = data_loss(u_hat, u_true, reduction="mean")
    return ldata + alpha * lpde + beta * lmono, ldata
import torch
from src.models import FNet, GNet
from src.losses import gradients, pde_loss, mono_loss, total_loss

# models
fnet = FNet()
gnet = GNet()

# optimizer (updates both nets)
opt = torch.optim.Adam(list(fnet.parameters()) + list(gnet.parameters()), lr=1e-3)

# fake batch (we'll replace with real dataset later)
B = 8
t = torch.rand(B, 1, requires_grad=True)
x = torch.rand(B, 16, requires_grad=True)
u_true = torch.rand(B, 1)  # fake labels

# forward
u_hat = fnet(t, x)
u_t, u_x = gradients(u_hat, t, x)

lpde = pde_loss(gnet, t, x, u_hat, u_t, u_x)
lmono = mono_loss(u_hat)  # placeholder; true mono needs sequences

L, ldata = total_loss(u_hat, u_true, lpde, lmono, alpha=1.0, beta=1.0)

# backward + update (THIS is the core training ritual)
opt.zero_grad()
L.backward()
opt.step()

print("L_total:", float(L), "L_data:", float(ldata), "L_PDE:", float(lpde), "L_mono:", float(lmono))

