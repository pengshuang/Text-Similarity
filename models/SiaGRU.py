import torch
import torch.nn as nn


class SiaGRU(nn.Module):
    def __init__(self, args):
        super(SiaGRU, self).__init__()
        self.args = args
        self.embeds_dim = args.embeds_dim
        num_word = 20000
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.ln_embeds = nn.LayerNorm(self.embeds_dim)
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden, cell = self.gru(x)
        return hidden.squeeze()

    def forward(self, *input):
        # sent1: batch_size * seq_len
        sent1 = input[0]
        sent2 = input[1]

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.ln_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.ln_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        encoding1 = self.forward_once(x1)
        encoding2 = self.forward_once(x2)

        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        return self.fc(sim)


