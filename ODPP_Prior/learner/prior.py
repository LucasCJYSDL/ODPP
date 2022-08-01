import torch.nn as nn
from torch.distributions.categorical import Categorical

from learner.base_mlp import MLP


class Prior(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, is_high=False):
        super(Prior, self).__init__()
        self.is_high = is_high
        self.mlp = MLP(layers=[input_dim, hidden_dim, hidden_dim, code_dim])

    def forward(self, x, code_gt=None):
        logits = self.mlp(x)
        if not self.is_high: # the last dim is saved for the primitive action, TODO: danger
            code_dist = Categorical(logits=logits[:, :-1])
        else:
            code_dist = Categorical(logits=logits)
        code = code_dist.sample() # (bs, )

        self.code_dist = code_dist
        logp_code = code_dist.log_prob(code).squeeze() # (bs, ), check
        if code_gt is not None:
            logp = code_dist.log_prob(code_gt).squeeze()
        else:
            logp = None

        return code, logp, logp_code

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    test = Prior(2, 64, 10)
    test_seq = torch.ones(size=(180, 2))  # (bs, s_dim)
    c, _, _ = test(test_seq)
    c_onehot = F.one_hot(c, 10).float()
    print(c.shape, c_onehot.shape)