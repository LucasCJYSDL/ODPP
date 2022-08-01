import torch.nn as nn
from torch.distributions.categorical import Categorical

from learner.base_mlp import MLP

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim):
        super(Decoder, self).__init__()
        self.mlp = MLP(layers=[input_dim, hidden_dim, hidden_dim, code_dim])

    def forward(self, seq, gt=None): # (bs, s_dim)
        self.logits = self.mlp(seq) # (bs, c_dim)
        policy = Categorical(logits=self.logits)
        code = policy.sample() # (bs, )
        logp = policy.log_prob(code).squeeze() # (bs, )
        if gt is not None:
            loggt = policy.log_prob(gt).squeeze()
        else:
            loggt = None

        return code, loggt, logp


if __name__ == '__main__':
    import torch
    test = Decoder(2, 64, 10)
    test_seq = torch.zeros(size=(180, 2))  # (bs, s_dim)
    test(test_seq)