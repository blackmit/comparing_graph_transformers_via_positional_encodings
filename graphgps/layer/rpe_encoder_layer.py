""" Layer for encoding the RPE matrix to be used as bias for attention

Modified from https://github.com/lsj2408/Graphormer-GD/blob/master/graphormer/modules/graphormer_layers.py
"""
import torch
import torch_geometric
import torch.nn.functional as F
# Permutes from (batch, node, node, head) to (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(torch.nn.Module):
    def __init__(self, num_rpes, num_kernels=128):
        super().__init__()
        self.num_rpes = num_rpes
        self.num_kernels = num_kernels
        self.means = torch.nn.Embedding(num_rpes, num_kernels)
        self.stds = torch.nn.Embedding(num_rpes, num_kernels)
        torch.nn.init.uniform_(self.means.weight, 0, 3)
        torch.nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x):
        x = x.unsqueeze(-1) # add a dimension to the end
        x = x.expand(-1, -1, -1, -1, self.num_kernels) # copy the tensor along the final axis K times
        mean = self.means.weight.float() 
        std = self.stds.weight.float().abs() + 1e-2
        return gaussian(x, mean, std).type_as(self.means.weight).flatten(3)
    
class NonLinear(torch.nn.Module):
    """ Two-layer MLP for encoding distance matrices """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NonLinear, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

class RPEEncoderLayer(torch.nn.Module):
    def __init__(self, 
        num_kernels : int, 
        num_heads : int,
        num_rpes : int,
        hidden_dim : int,
        bias_type : str, 
        use_gaussians : bool
    ):
        """ Layer for encoding RPEs for RPE Transformer

        Args:
            num_kernels: number of Gaussian kernels
            num_heads: number of Transformer heads
            num_rpes: Number of different RPEs
            bias type: The type of bias ("add" or "mult")
                The bias is saved as an attribute `{bias_type}_bias` of the data
            use_gaussian: whether or not to use Gaussian kernels
        """
        super().__init__()
        if bias_type != "add" and bias_type != "mult":
            raise ValueError("Distance encoder layer must be of type add or mult")
        self.bias_type = bias_type
        self.use_gaussians = use_gaussians
        if self.use_gaussians:
            self.gbf = GaussianLayer(num_rpes, num_kernels)
            self.mlp = NonLinear(num_rpes*num_kernels, num_heads, hidden_dim)
        else:
            self.mlp = NonLinear(num_rpes, num_heads, hidden_dim)


    def reset_parameters(self):
        pass

    def forward(
        self,
        data : torch_geometric.data.Batch
    ) -> torch.Tensor:
        rpe = data.attn_bias
        if self.use_gaussians:
            bias = self.mlp(self.gbf(rpe.float())).permute(BATCH_HEAD_NODE_NODE)
        else:
            bias =  self.mlp(rpe.float()).permute(BATCH_HEAD_NODE_NODE)
        if self.bias_type == "add":
            data.add_bias = bias 
        elif self.bias_type == "mult":
            data.mult_bias = bias
        return data
