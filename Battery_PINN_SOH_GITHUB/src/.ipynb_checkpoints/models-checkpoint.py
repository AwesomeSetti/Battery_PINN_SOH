import torch
import torch.nn as nn

class FNet(nn.Module):
    def __init__(self, in_dim=17, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, x):
        # t: (B, 1), x: (B, 16)
        inp = torch.cat([t, x], dim=1)  # -> (B, 17)
        return self.net(inp)
class GNet(nn.Module):
    def __init__(self, in_dim=35, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z)
