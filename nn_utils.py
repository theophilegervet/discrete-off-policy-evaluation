import math

import torch
import torch.nn as nn


class EnsembleLinear(nn.Module):
    def __init__(self, ensemble_size, input_dim, output_dim):
        """Applies a linear transformation to the incoming data.
        
        :param ensemble_size: number of ensemble members
        """
        super(EnsembleLinear, self).__init__()
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, input_dim, output_dim))
        self.bias = nn.Parameter(
            torch.Tensor(ensemble_size, 1, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, input):
        """
        :param input: torch.Tensor (ensemble_size, batch_size, input_dim)
        :returns: torch.Tensor (ensemble_size, batch_size, output_dim)
        """
        return input.matmul(self.weight) + self.bias