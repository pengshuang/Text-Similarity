import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)

