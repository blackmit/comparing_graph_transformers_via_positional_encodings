
import torch
from typing import Optional

class MLP(torch.nn.Module):
    """ Multi-layer perceptron """
    def __init__(
        self, 
        input_dim : int, 
        hidden_dim: int,
        output_dim : int, 
        num_layers : int,
    ):
        self.output_dim = output_dim
        super(MLP, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers-2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            return self.layers(x)
        else:
            # if x is not a 2-dimensional tensor,
            # we need to reshape it so that it is compatible 
            # with batch normaliztion
            shape = list(x.shape)
            shape[-1] = self.output_dim
            x = x.flatten(end_dim=-2)
            x = self.layers(x)
            x = x.reshape(shape)
            return x
