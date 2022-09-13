import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np

from learner.base_mlp import MLP

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, output_activation=None, act_range=None):
        super(GaussianPolicy, self).__init__()

        self.mu = MLP(layers=[input_dim, hidden_dim, hidden_dim, action_dim], output_activation=output_activation)
        # self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim)) # TODO: check how usually this works
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act_range = act_range

    def forward(self, x, a=None):
        policy = Normal(self.act_range * self.mu(x), self.log_std.exp())
        pi = policy.sample() # (bs, action_dim)
        logp_pi = policy.log_prob(pi).sum(dim=1) # multiplication -> joint pdf across the dimensions of the action space, log of pdf (always > 0), (bs, )
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        return pi, logp, logp_pi

class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(CategoricalPolicy, self).__init__()

        self.mlp = MLP(layers=[input_dim, hidden_dim, hidden_dim, action_dim])

    def forward(self, x, a=None):
        logits = self.mlp(x)
        policy = Categorical(logits=logits)
        pi = policy.sample() # (bs, )
        logp_pi = policy.log_prob(pi).squeeze() # (bs, ), check
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi


if __name__ == '__main__':
    # test = GaussianPolicy(2, 64, 10)
    test = CategoricalPolicy(2, 64, 10)
    test_seq = torch.zeros(size=(180, 2))  # (bs, s_dim)
    test(test_seq)