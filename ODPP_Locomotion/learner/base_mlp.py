import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None, init=True): # TODO：relu
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            if init:
                nn.init.zeros_(self.layers[i].bias) # TODO: check if the initialization matters
        # print(self.layers)
    #
    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        #self.feature = x.detach().clone()
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x

if __name__ == '__main__':
    test = MLP([2, 64, 64, 10], activation=torch.nn.functional.relu)
    test_seq = torch.zeros(size=(180, 2)) # (bs, s_dim)
    print(test(test_seq).shape)
    print(test.feature.shape)
