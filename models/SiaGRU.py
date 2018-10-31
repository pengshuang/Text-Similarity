from .modules import Swish
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiaGRU(nn.Module):
    def __init__(self, args):
        super(SiaGRU, self).__init__()
        self.args = args
        self.embeds_dim = args.embeds_dim
        num_word = 20000
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.hidden_size = args.hidden_size
        self.linear_size = args.linear_size
        self.num_layer = args.num_layer

        self.gru = nn.ModuleList([nn.GRU(self.embeds_dim, self.hidden_size, bidirectional=True)] +
                                 [nn.GRU(2 * self.hidden_size, self.hidden_size, bidirectional=True) for _ in range(self.num_layer-1)])
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))
        self.ln = nn.ModuleList([nn.LayerNorm(self.embeds_dim)] +
                                [nn.LayerNorm(self.hidden_size) for _ in range(2 * self.num_layer)])

        # self.fc = nn.Sequential(
        #     nn.Linear((2*self.hidden_size)*(1+self.num_layer)*2, self.linear_size),
        #     nn.LayerNorm(self.linear_size),
        #     Swish(),
        #     nn.Linear(self.linear_size, 2),
        #     nn.Softmax()
        # )
        self.fc1 = nn.Sequential(
            nn.Linear((2*self.hidden_size)*(1+self.num_layer)*2, self.linear_size),
            nn.LayerNorm(self.linear_size),
            Swish(),
            nn.Linear(self.linear_size, 1),
            # nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear((2*self.hidden_size)*(1+self.num_layer), self.embeds_dim),
        )
        self.fc3 = nn.Sequential(
            nn.Linear((2*self.hidden_size)*(1+self.num_layer), self.embeds_dim),
        )
        self.fc = nn.Sequential(
            # nn.Linear(3, 50),
            # nn.LayerNorm(50),
            # Swish(),
            nn.BatchNorm1d(3),
            nn.Sigmoid(),
            nn.Linear(3, 2),
            nn.Softmax()
        )
        self.l3 = nn.Linear(1, 1)

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward(self, *input):
        sent1 = input[0]
        sent2 = input[1]
        x = torch.cat([sent1, sent2], 0)
        x = self.embeds(x).permute(1, 0, 2)
        x = self.ln[0](x)
        hiddens = []
        for layer, rnn in enumerate(self.gru):
            o, hn = rnn(x, self.h0[2*layer:2*(layer+1), :, :].repeat(1, x.size(1) ,1))
            hiddens.append(hn.transpose(0, 1).contiguous().view(hn.size(1), -1))
            x_forward = self.ln[2*layer+1](o[:,:,:self.hidden_size])
            x_backward = self.ln[2*layer+2](o[:,:,self.hidden_size:])
            x = torch.cat([x_forward, x_backward], -1)

        output = F.max_pool1d(x.permute(1, 2, 0), x.size(0)).squeeze(2)
        x = torch.cat(hiddens + [output], 1)
        # x = torch.cat([x[:x.size(0)//2,:], x[x.size(0)//2:,:]], 1)
        # sim = self.fc(x)
        # return sim
        # new
        x1, x2 = x[:x.size(0)//2,:], x[x.size(0)//2:,:]
        sim1 = self.fc1(torch.cat([x1, x2], 1))
        sim2 = torch.norm(self.fc2(x1)-self.fc2(x2), p=2, dim=-1, keepdim=True)
        sim3 = self.l3(F.cosine_similarity(self.fc3(x1), self.fc3(x2)).unsqueeze(1))
        # import ipdb; ipdb.set_trace()
        sim = torch.cat([sim1, sim2, sim3], -1)
        return self.fc(sim)

        # 18057145962


