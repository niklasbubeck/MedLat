import torch
import numpy as np

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, channel_dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=channel_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.tensor([0.0], device=self.mean.device)
        else:
            reduce_dims = list(range(1, self.mean.ndim))  # all dims except batch
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dims
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=reduce_dims
                )

    def nll(self, sample):
        if self.deterministic:
            return torch.tensor([0.0], device=self.mean.device)
        logtwopi = np.log(2.0 * np.pi)
        reduce_dims = list(range(1, self.mean.ndim))  # sum over everything but batch
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=reduce_dims
        )

    def mode(self):
        return self.mean
    