import numpy as np
import torch
from torch.autograd import Variable


class DiagNormal:
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = torch.exp(logstd)

    def sample(self, sample_shape=torch.Size()):
        eps = Variable(torch.randn(sample_shape)).cuda()
        return self.mean + eps * self.std

    def neglogp(self, x):
        return 0.5 * torch.sum(torch.pow((x - self.mean) / (self.std + 1e-8),
                                         2.0), dim=-1) \
               + 0.5 * np.log(2.0 * np.pi) * float(x.size()[-1]) \
               + torch.sum(self.logstd, dim=-1)

    def entropy(self):
        return torch.sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e),
                         dim=-1)

    def kl(self, other):
        return torch.sum(other.logstd - self.logstd +
                         (torch.pow(self.std, 2) +
                          torch.pow(self.mean - other.mean, 2)) /
                         (2.0 * torch.pow(other.std, 2)) - 0.5, dim=-1)
