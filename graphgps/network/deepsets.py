import torch

from graphgps.layer.ign_layer import layer_1_to_1

class DeepSets(torch.nn.Module):
    def __init__(
        self, 
        input_dim : int,  
        hidden_dim : int,
        output_dim : int, 
        num_layers : int,
    ):
        super(DeepSets, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        layers = []
        layers.append(layer_1_to_1(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers-1):
            layers.append(layer_1_to_1(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
        layers.append(layer_1_to_1(hidden_dim, output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.layers(X)