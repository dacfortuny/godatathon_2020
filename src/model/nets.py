import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=True)

    def forward(self, x):
        # Initialize hidden with zeros
        batch_size = x.shape[1]
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim,
                         device=x.device)
        output, hn = self.rnn(x, h0)
        return output, hn
