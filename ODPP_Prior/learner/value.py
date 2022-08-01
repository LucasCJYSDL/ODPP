import torch.nn as nn
from learner.base_mlp import MLP

# TODO: share layer with the policy netwrok
class ValueFuntion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueFuntion, self).__init__()

        self.mlp = MLP(layers=[input_dim, hidden_dim, hidden_dim, 1])

    def forward(self, x): # (bs, s_dim)
        v = self.mlp(x) # (bs, 1)
        return v