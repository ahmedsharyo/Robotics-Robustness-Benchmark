import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=8, action_dim=2):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, action_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def distribution(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            dist = self.distribution(x)
            a = dist.mean if deterministic else dist.sample()
            return a.clamp(-1,1).cpu().numpy()
    def forward(self, obs): return self.actor(obs), self.critic(obs).squeeze(-1)


def load_policy(path, device="cpu"):
    model = ActorCritic().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
