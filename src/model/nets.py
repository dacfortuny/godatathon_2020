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
                         dtype=x.dtype, device=x.device)

        output, hn = self.rnn(x, h0)
        return output, hn


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, h0):
        h0 = self.dropout(h0)
        output, hn = self.rnn(x, h0)
        prediction = self.fc(output)

        return prediction, hn


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Seq2Seq, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(1, hidden_dim, num_layers)

    def forward(self, x, y=None):
        y_length = 24  # Number of months to predict
        batch_size = x.shape[1]

        _, encoder_hidden_out = self.encoder(x)

        predictions = torch.zeros(y_length, batch_size).to(x.device)

        # Take data from month -1 as first input for decoder
        decoder_input = x[[-1]]

        # TODO: Add compatibility for passing time data
        # Select only the first channel (volume)
        decoder_input = decoder_input[:, :, [0]]

        decoder_hidden = encoder_hidden_out

        for i in range(y_length):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            predictions[i] = decoder_out.flatten()

            # TODO: Add teacher forcing
            decoder_input = decoder_out

        return predictions.unsqueeze(dim=-1)
