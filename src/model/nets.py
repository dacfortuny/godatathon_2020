import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=True)

    def forward(self, temp_features, static_features):
        h0 = torch.stack(2 * self.num_layers * [static_features])
        output, hn = self.rnn(temp_features, h0)
        return output, hn


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 3)

    def forward(self, x, h0):
        h0 = self.dropout(h0)
        output, hn = self.rnn(x, h0)
        prediction = self.fc(output)

        return prediction, hn


class Seq2Seq(nn.Module):
    EMBEDDING_DIM = 4

    def __init__(self, input_dim, hidden_dim, num_layers,
                 n_countries=16, n_brands=484, n_packages=7, n_therapeutical=14):
        super(Seq2Seq, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.country_emb = nn.Embedding(num_embeddings=n_countries,
                                        embedding_dim=self.EMBEDDING_DIM)
        self.brand_emb = nn.Embedding(num_embeddings=n_brands,
                                      embedding_dim=self.EMBEDDING_DIM)
        self.package_emb = nn.Embedding(num_embeddings=n_packages,
                                        embedding_dim=self.EMBEDDING_DIM)
        self.therapeutical_emb = nn.Embedding(num_embeddings=n_therapeutical,
                                              embedding_dim=self.EMBEDDING_DIM)

        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(1, hidden_dim, num_layers)

    def forward(self, temp_features, num_features, cat_features, y=None):
        y_length = 24  # Number of months to predict
        batch_size = temp_features.shape[1]

        # Embeddings
        categorical_emb_features = torch.cat(
            [
                self.country_emb(cat_features[0]),
                self.brand_emb(cat_features[1]),
                self.package_emb(cat_features[2]),
                self.therapeutical_emb(cat_features[3])
            ], dim=1)

        # Static features (not changing over time)
        static_features = torch.cat([num_features, categorical_emb_features], dim=1)

        _, encoder_hidden_out = self.encoder(temp_features, static_features)

        predictions = torch.zeros(y_length, batch_size, 1).to(temp_features.device)
        upper_bounds = torch.zeros(y_length, batch_size, 1).to(temp_features.device)
        lower_bounds = torch.zeros(y_length, batch_size, 1).to(temp_features.device)

        # Take data from month -1 as first input for decoder
        decoder_input = temp_features[[-1]]

        # Select only the first channel (volume)
        decoder_input = decoder_input[:, :, [0]]

        decoder_hidden = encoder_hidden_out

        for i in range(y_length):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Note: decoder_out: [1, bs, 3] -> 3:(prediction, upper_bound, lower_bound)
            decoder_input = decoder_out[:, :, [0]]

            predictions[i] = decoder_out[0, :, [0]]
            upper_bounds[i] = decoder_out[0, :, [1]]
            lower_bounds[i] = decoder_out[0, :, [2]]

            # TODO: Add teacher forcing

        return {"prediction": predictions,
                "upper_bound": upper_bounds,
                "lower_bound": lower_bounds}
