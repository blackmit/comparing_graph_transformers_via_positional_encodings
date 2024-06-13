from ..network.mlp import MLP
import torch
from torch_geometric.data import Batch
from torch_geometric.graphgym import cfg 
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_batch

def create_batch_mask(batch : torch.Tensor) -> torch.Tensor:
    """ Given a batch mask batch \in \Z^{n_tot} 
    where batch[i] = batch of node i,
    return a tensor mask \in {0,1}^{b x n_max x n_max} where 
    mask[b, i, j] indicates whether there is an ith and jth node in graph b
    """
    num_nodes_per_batch = torch.bincount(batch)
    mask = torch.zeros(len(num_nodes_per_batch), torch.max(num_nodes_per_batch), torch.max(num_nodes_per_batch))
    for graph, num_nodes in enumerate(num_nodes_per_batch):
        mask[graph,:num_nodes,:num_nodes] = 1
    return mask.bool()

def from_dense_adj(x : torch.Tensor, batch : torch.Tensor) -> torch.Tensor:
    """ Given a tensor of shape b x n_max x n_max (x dim), return a tensor of size n_tot (x dim) 
    
    This is the inverse of torch_geometric.utils.to_dense_adj
    """
    x_flattened = x.flatten(end_dim=2)
    mask = create_batch_mask(batch)
    mask_flattened = mask.flatten()
    if x.dim() == 3:
        return x_flattened[mask_flattened]
    else: 
        return x_flattened[mask_flattened,:]


@register_node_encoder("DeepSetsAPEEncoder")
class DeepSetsAPEEncoder(torch.nn.Module):
    def __init__(self, dim_emb, *args, **kwargs):
        super().__init__()
        self.mlp = MLP(
            cfg.deepsets_ape_encoder.input_dim,
            cfg.deepsets_ape_encoder.hidden_dim,
            cfg.deepsets_ape_encoder.output_dim,
            cfg.deepsets_ape_encoder.num_hidden_layers
        )
        self.ape_type = cfg.deepsets_ape_encoder.ape_type

    def forward(self, data : Batch):
        ape, _ = to_dense_batch(
            getattr(data, f"pestat_{self.ape_type}"), 
            batch=data.batch
        ) 
        transformed_ape = self.mlp(ape)
        rpe = transformed_ape.unsqueeze(-2) + transformed_ape.unsqueeze(-3)  
        rpe_flattened = from_dense_adj(rpe, data.batch)
        if hasattr(data, "rpe"):
            data.rpe = torch.column_stack((
                data.rpe,
                rpe_flattened
            )) 
        else:    
            data.rpe = rpe_flattened
        return data 


