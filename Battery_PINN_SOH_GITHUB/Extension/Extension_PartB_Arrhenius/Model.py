import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from util import AverageMeter, get_logger, eval_metrix
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, layers_num=4, hidden_dim=50, droupout=0.2):
        super().__init__()
        assert layers_num >= 2, "layers must be greater than 2"
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(Sin())
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(Sin())
                layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 32),
            Sin(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class Solution_u(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        self.encoder = MLP(input_dim=input_dim, output_dim=32, layers_num=3, hidden_dim=60, droupout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def forward(self, x):
        x = self.encoder(x)
        return self.predictor(x)

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {count} trainable parameters")


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group.get("name", "") == "predictor":
                param_group["lr"] = self.base_lr
            else:
                lr = param_group["lr"] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


class StressNet(nn.Module):
    """
    Optional s(cond) > 0 scaling net.
    We keep it tiny to avoid overfitting.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            Sin(),
            nn.Linear(16, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, c):
        # positive scale: s = 1 + softplus(...)
        return 1.0 + self.softplus(self.net(c))


class PINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        # ----------------------------
        # A1) Inputs (already done)
        # ----------------------------
        self.input_dim = int(getattr(args, "input_dim", 20))
        self.solution_u = Solution_u(input_dim=self.input_dim).to(device)

        # dynamical F: expects concat([xt, u, u_x, u_t])
        # xt dim = input_dim, u dim=1, u_x dim=(input_dim-1), u_t dim=1 => total = input_dim + 1 + (input_dim-1) + 1 = 2*input_dim + 1
        dyn_in_dim = 2 * self.input_dim + 1
        self.dynamical_F = MLP(
            input_dim=dyn_in_dim,
            output_dim=1,
            layers_num=args.F_layers_num,
            hidden_dim=args.F_hidden_dim,
            droupout=0.2,
        ).to(device)

        # ----------------------------
        # A2) Arrhenius settings
        # ----------------------------
        self.use_arrhenius = bool(getattr(args, "use_arrhenius", True))
        self.Tref = float(getattr(args, "Tref", 298.15))
        # With our enforced ordering: [..., temp_C, temp_K, condition_id, cycle]
        # temp_K is the 3rd from the end among features => xt[:, -3]
        self.tempK_col = int(getattr(args, "tempK_col", -3))

        # Learnable beta (kept positive via softplus)
        self.arr_beta_raw = nn.Parameter(torch.tensor(0.0, device=device))
        self.softplus = nn.Softplus()
        self.arr_eps = 1e-8

        # ----------------------------
        # A3) Optional condition scaling s(cond)
        # ----------------------------
        self.use_stressnet = bool(getattr(args, "use_stressnet", False))
        # simplest cond vector: [condition_id, temp_K] (you can expand later)
        # We'll default to 2 dims: condition_id and temp_K
        self.cond_cols = getattr(args, "cond_cols", None)  # e.g. "-2,-3"
        if self.use_stressnet:
            self.cond_dim = int(getattr(args, "cond_dim", 2))
            self.stress_net = StressNet(in_dim=self.cond_dim).to(device)
        else:
            self.stress_net = None

        # ----------------------------
        # Optimizers
        # ----------------------------
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)

        opt2_params = list(self.dynamical_F.parameters()) + [self.arr_beta_raw]
        if self.use_stressnet:
            opt2_params += list(self.stress_net.parameters())

        self.optimizer2 = torch.optim.Adam(opt2_params, lr=args.lr_F)

        self.scheduler = LR_Scheduler(
            optimizer=self.optimizer1,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr,
        )

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        self.best_model = None
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def predict(self, xt):
        return self.solution_u(xt)

    def _arrhenius_kT(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Relative Arrhenius multiplier:
            k(T) = exp(beta * (1/Tref - 1/T))
        beta is learned (positive).
        """
        T = xt[:, self.tempK_col:self.tempK_col + 1]
        T = torch.clamp(T, min=1.0)
        beta = self.softplus(self.arr_beta_raw) + self.arr_eps
        invT = 1.0 / T
        invTref = 1.0 / self.Tref
        return torch.exp(beta * (invTref - invT))

    def _cond_vector(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Build cond vector for StressNet.
        Default: [condition_id, temp_K] using columns [-2] and [-3] given our ordering.
        """
        if self.cond_cols is None:
            c = torch.cat([xt[:, -2:-1], xt[:, -3:-2]], dim=1)  # [cond_id, temp_K]
            return c
        # custom: cond_cols like "-2,-3"
        idxs = [int(s.strip()) for s in str(self.cond_cols).split(",")]
        cols = [xt[:, i:i + 1] for i in idxs]
        return torch.cat(cols, dim=1)

    def forward(self, xt):
        xt.requires_grad = True

        # x = all features except time
        x = xt[:, 0:-1]
        # t = last feature (cycle)
        t = xt[:, -1:]

        u = self.solution_u(torch.cat((x, t), dim=1))

        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True, allow_unused=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True, allow_unused=True)[0]

        # Dynamics net
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))

        # ----------------------------
        # A2/A3) Physics residual
        # ----------------------------
        rate_scale = 1.0

        # Arrhenius scaling
        if self.use_arrhenius:
            rate_scale = rate_scale * self._arrhenius_kT(xt)

        # Condition scaling s(cond)
        if self.use_stressnet:
            c = self._cond_vector(xt)
            rate_scale = rate_scale * self.stress_net(c)

        # PDE residual
        f = u_t - rate_scale * F
        return u, f

    def Test(self, testloader):
        self.eval()
        true_label, pred_label = [], []
        with torch.no_grad():
            for _, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label

    def Valid(self, validloader):
        self.eval()
        true_label, pred_label = [], []
        with torch.no_grad():
            for _, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()

        for it, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)

            # data loss
            loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5 * self.loss_func(f1, f_target) + 0.5 * self.loss_func(f2, f_target)

            # monotonic / physics loss (prevents capacity regeneration)
            loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()

            loss = loss1 + self.alpha * loss2 + self.beta * loss3

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())

            if (it + 1) % 50 == 0:
                print(f"[epoch:{epoch} iter:{it+1}] data loss:{loss1:.6f}, PDE loss:{loss2:.6f}, physics loss:{loss3:.6f}")

        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0

        for e in range(1, self.args.epochs + 1):
            early_stop += 1
            loss1, loss2, loss3 = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()

            info = f"[Train] epoch:{e}, lr:{current_lr:.6f}, total loss:{loss1 + self.alpha*loss2 + self.beta*loss3:.6f}"
            self.logger.info(info)

            if validloader is not None:
                valid_mse = self.Valid(validloader)
                self.logger.info(f"[Valid] epoch:{e}, MSE: {valid_mse}")

            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                MAE, MAPE, MSE, RMSE = eval_metrix(pred_label, true_label)
                self.logger.info(f"[Test] MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}")
                early_stop = 0

                # save best model
                self.best_model = {
                    "solution_u": self.solution_u.state_dict(),
                    "dynamical_F": self.dynamical_F.state_dict(),
                    "arr_beta_raw": self.arr_beta_raw.detach().cpu().numpy(),
                }
                if self.use_stressnet:
                    self.best_model["stress_net"] = self.stress_net.state_dict()

                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, "true_label.npy"), true_label)
                    np.save(os.path.join(self.args.save_folder, "pred_label.npy"), pred_label)

            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                self.logger.info(f"early stop at epoch {e}")
                break

        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, "model.pth"))
