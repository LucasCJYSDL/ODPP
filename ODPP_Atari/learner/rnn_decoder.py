import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class RNN_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim):
        super(RNN_Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, code_dim)
        nn.init.zeros_(self.linear.bias)

    # TODO: check how Bi-LSTM usually works
    def forward(self, seq, gt=None): # (bs, epi_len, s_dim)
        self.inter_states, _ = self.lstm(seq) # (bs, epi_len, 2 * hidden_dim) # bi-lstm
        logit_seq = self.linear(self.inter_states) # (bs, epi_len, c_dim) # linear
        self.logits = torch.mean(logit_seq, dim=1) # (bs, c_dim) # average pooling
        policy = Categorical(logits=self.logits)
        code = policy.sample() # (bs, )
        logp = policy.log_prob(code).squeeze() # (bs, )
        if gt is not None:
            loggt = policy.log_prob(gt).squeeze()
        else:
            loggt = None

        return code, loggt, logp

if __name__ == '__main__':
    test_decoder = RNN_Decoder(2, 64, 10)
    test_seq = torch.zeros(size=(180, 20, 2)) # (bs, epi_len, s_dim)
    test_decoder(test_seq)