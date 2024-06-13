from ..layer.ign_layer import layer_2_to_2, layer_2_to_1
import torch
from torch_geometric.data import Batch
from torch_geometric.graphgym import cfg 
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_adj

# Permutes from (batch, node, node, dim) to (batch, dim, node, node)
BATCH_DIM_NODE_NODE = (0, 3, 1, 2)
# Permutes from (batch, dim, node) to (batch, node, dim)
BATCH_NODE_DIM = (0, 2, 1)

def create_batch_mask(batch : torch.Tensor) -> torch.Tensor:
    """ Given a batch mask batch \in \Z^{n_tot} 
    where batch[i] = batch of node i,
    return a tensor mask \in {0,1}^{b x n_max} where 
    mask[j, i] indicates whether there is an ith node in graph j
    """
    num_nodes_per_batch = torch.bincount(batch)
    mask = torch.zeros(len(num_nodes_per_batch), torch.max(num_nodes_per_batch))
    for graph, num_nodes in enumerate(num_nodes_per_batch):
        mask[graph,:num_nodes] = 1
    return mask.bool()
    
def from_dense_batch(x : torch.Tensor, batch : torch.Tensor) -> torch.Tensor:
    """ Given a tensor of shape b x n_max x dim, return a tensor of size n_tot x dim 

    This is the inverse of torch_geometric.utils.to_dense_batch
    """
    x_flattened = x.flatten(end_dim=1)
    mask = create_batch_mask(batch)
    mask_flattened = mask.flatten()
    return x_flattened[mask_flattened,:]

class IGNEncoder(torch.nn.Module):
    def __init__(
        self, 
        in_dim : int , 
        hidden_dim : int, 
        out_dim : int,
        num_layers : int, 
        type : str,
    ):
        super().__init__()
        self.type = type
        if type == "sum":
            if out_dim != cfg.rpe_transformer.embed_dim:
                raise ValueError(f"cfg.ign_encoder.out_dim",
                                 "and cfg.rpe_transformer.embed_dim must match",
                                 "when using ign_encoder.type = add ")
        elif self.type == "concatenation":
            # TODO : add dimension test
            pass
        else:
            raise ValueError(f"Unsupported ign_encoder.type {self.type}")  
        layers = []
        layers.append(layer_2_to_2(in_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm2d(hidden_dim))
        for _ in range(num_layers-1):
            layers.append(layer_2_to_2(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm2d(hidden_dim))
        layers.append(layer_2_to_1(hidden_dim, out_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm1d(out_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch : Batch) -> Batch:
        """ Pass the rpe in `batch` through an IGN and add or concatenate it to the node features  """
        rpe = batch.attn_bias.permute(BATCH_DIM_NODE_NODE)
        ape = self.layers(rpe.float()).permute(BATCH_NODE_DIM)
        ape_flattened = from_dense_batch(ape, batch.batch)
        if self.type == "concatenation":
            batch.x = torch.cat((batch.x, ape_flattened), 1)
        elif self.type == "sum":
            batch.x += ape_flattened
        return batch

@register_node_encoder('IGNRPEEncoder')
class IGNRPEEncoder(torch.nn.Sequential):
    def __init__(self, dim_emb, *args, **kwargs):
        encoder = IGNEncoder(
            in_dim=cfg.posenc_RPE.num_rpes,
            hidden_dim=cfg.ign_encoder.hidden_dim,
            out_dim=cfg.ign_encoder.out_dim,
            num_layers=cfg.ign_encoder.num_hidden_layers,
            type=cfg.ign_encoder.type,
        )
        super().__init__(encoder)



