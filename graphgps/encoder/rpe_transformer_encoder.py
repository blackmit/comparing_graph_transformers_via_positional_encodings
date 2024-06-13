import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_networkx, get_laplacian

from graphgps.network.deepsets import DeepSets

# Permutes from (batch, node, node, head) to (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)
# Permutes from (batch, node, dim) to (batch, dim, node)
BATCH_DIM_NODE = (0, 2, 1)
# Permutes from (batch, dim, node) to (batch, node, dim)
BATCH_NODE_DIM = (0, 2, 1)
# Inserts a leading 0 row and a leading 0 column with F.pad
INSERT_GRAPH_TOKEN = (0, 0, 1, 0, 1, 0)

def set_intercomponent_distance_to_inf(
    distance_matrix : torch.Tensor,
    component_mask : torch.Tensor,
    inf_distance : float
):
    if distance_matrix.dim() == 4:
        component_mask = component_mask.unsqueeze(-1)
    neg_component_mask = torch.logical_not(component_mask)
    distance_matrix = component_mask*distance_matrix + inf_distance*neg_component_mask
    return distance_matrix

def convert_gram_matrix_to_distance(
    gram_matrix : torch.Tensor
) -> torch.Tensor:
    """ Convert the (Gram) matrix of a kernel into the corresponding squared distance matrix
     
    This assumes the input is a Gram matrix
    of some point set that is centered at the origin.
    it returns the distance matrix of that point set.
    Note that all graph spectral kernels
    are Gram matrices of a centered point set.
    
    The input matrix is of size b x n x n (x d).
    The output matrix is of size b x n x n (x d).

    """
    if gram_matrix.dim() == 3:
        gram_matrix_diagonal = torch.diagonal(gram_matrix, dim1=1, dim2=2)
    else:
        # torch.diagonal appends the diagonal to the last dimension 
        # so in the case of 4-dimensional tensors,
        # we need to reshape the output so the diagonal terms are on the 1-axis
        gram_matrix_diagonal = torch.diagonal(gram_matrix, dim1=1, dim2=2).permute((0,2,1))
    # This one-liner is using the formula
    #   distance_matrix[i,j,k,l] = gram_matrix[i,j,j,l] + gram_matrix[i,k,k,l] -2*gram_matrix[i,j,k,l]
    distance_matrix = \
        gram_matrix_diagonal.unsqueeze(1) \
        + gram_matrix_diagonal.unsqueeze(2) \
        - 2*gram_matrix
    return distance_matrix

def create_batch_mask(batch : torch.Tensor) -> torch.Tensor:
    """ Given a batch mask `batch` \in \Z^{n_tot} 
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
    """ Given a tensor of shape b x n_max x n_max (x dim), return a tensor of size n_tot (x dim) """
    x_flattened = x.flatten(end_dim=2)
    mask = create_batch_mask(batch)
    mask_flattened = mask.flatten()
    if x.dim() == 3:
        return x_flattened[mask_flattened]
    else: 
        return x_flattened[mask_flattened,:]


class NonLinear(torch.nn.Module):
    """ Two-layer MLP for encoding distance matrices """
    def __init__(self, input_dim, output_dim, hidden_dim=None):
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


class SPELayer(torch.nn.Module):
    def __init__(
        self, 
        hidden_dim : int,
        output_dim : int,
        num_layers : int,
        inf_distance : float,
        as_distance : bool,
    ):
        super().__init__()
        self.deepsets = DeepSets(
            input_dim=1,           # each eigenvalue is handle as its own token
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            num_layers=num_layers,
        )
        self.as_distance = False
        self.inf_distance = inf_distance
        self.as_distance = as_distance
        
    def forward(
        self,
        data : Batch,
    ) -> Batch:
        if not hasattr(data, "eigvecs") or not hasattr(data, "eigvals"):
            raise AttributeError(
                "Data has no attribute eigvals or eigvecs."
                " Data must first be preprocessed with the function add_laplacian_eigenpairs_to_data"
                " before SPELayer can be used."
            )
        eigvecs_matrix = to_dense_adj(
            data.complete_graph_index,
            batch=data.batch,
            edge_attr=data.eigvecs
        )
        eigvals, _ = to_dense_batch(
            data.eigvals,
            batch=data.batch
        )
        eigvals = eigvals.unsqueeze(-1) 
        transformed_eigvals = self.deepsets(eigvals.permute(BATCH_DIM_NODE).float()).permute(BATCH_NODE_DIM)
        # perform the outer product of the eigenvectors with transformed eigenvalues across different dimensions
        spe_kernels =  torch.einsum(
            "bne,bNed->bnNd", # b - batch, n - nodes, N - nodes (alt), e - eigenvalues, d - output dim
            eigvecs_matrix, 
            torch.einsum("bne,bed->bned", eigvecs_matrix, transformed_eigvals)
        )
        if self.as_distance:
            rpe = convert_gram_matrix_to_distance(spe_kernels)
            rpe = set_intercomponent_distance_to_inf(
                distance_matrix=rpe,
                component_mask=to_dense_adj(data.component_index, batch=data.batch),
                inf_distance=self.inf_distance
            )
        else:
            rpe = spe_kernels
        rpe_flattened = from_dense_adj(rpe, data.batch)
        if hasattr(data, "rpe"):
            data.rpe = torch.column_stack((
                data.rpe,
                rpe_flattened
            )) 
        else:    
            data.rpe = rpe_flattened
        return data 

class LearnableSpectralDistanceLayer(torch.nn.Module):
    """ Add a learnable diffusion matrix to a torch_geometric data batch.

        The learnable distance is a generalization of the resistance or diffusion distances.
        For eigenvectors and eigenvalues $x_i$ and $\lambda_i$ of the Laplacian,
        the learnable distance between vertices $u$ and $v$ is defined
        $$
            \sum_{i=2}^{n} \phi(\lambda_i) (1_s-1_t)^{T}x_ix_i^{T}(1_s-1_t),
        $$
        where $\phi:\mathbb{R}\to\mathbb{R}$ is a learnable function. 
        For this specific class, $\phi$ is a 2-layer mlp.
    """
    def __init__(
        self, 
        hidden_dim : int,
        inf_distance : float = 512
    ):
        super().__init__()
        self.inf_distance = inf_distance
        self.mlp = NonLinear(1, 1, hidden_dim)

    def forward(
        self,
        data : Batch,
    ) -> Batch:
        if not hasattr(data, "eigvecs") or not hasattr(data, "eigvals"):
            raise AttributeError(
                "Data has no attribute eigvals or eigvecs."
                " Data must first be preprocessed with the function add_laplacian_eigenpairs_to_data"
                " before LearnableDiffusionDistanceLayer can be used."
            )
        eigvecs_matrix = to_dense_adj(
            data.complete_graph_index,
            batch=data.batch,
            edge_attr=data.eigvecs
        )
        eigvals, real_nodes = to_dense_batch(
            data.eigvals,
            batch=data.batch
        )
        eigvals = eigvals.unsqueeze(-1)
        transformed_eigvals = self.mlp(eigvals.float())
        eigvals_diag = torch.diag_embed(transformed_eigvals[:,:,0]).double()
        eigvecs_matrix_T = eigvecs_matrix.permute((0, 2, 1))
        exp_laplacian = torch.bmm(torch.bmm(eigvecs_matrix, eigvals_diag), eigvecs_matrix_T)
        exp_laplacian_diagonal = torch.diagonal(exp_laplacian, dim1=1, dim2=2)
        distance_matrix = torch.sqrt(
            exp_laplacian_diagonal.unsqueeze(1) \
            + exp_laplacian_diagonal.unsqueeze(2) \
            - 2*exp_laplacian
        )
        # set distance between nodes in different connected components to self.inf_distance
        component_mask = to_dense_adj(
            data.component_index, 
            batch=data.batch
        )
        neg_component_mask = torch.logical_not(component_mask)
        distance_matrix = component_mask*distance_matrix + self.inf_distance*neg_component_mask
        distance_matrix_flattened = from_dense_adj(distance_matrix, data.batch)
        if hasattr(data, "rpe"):
            data.rpe = torch.column_stack((
                data.rpe,
                distance_matrix_flattened
            )) 
        else:    
            data.rpe = distance_matrix_flattened
        return data  

class LearnableDiffusionDistanceLayer(torch.nn.Module):
    def __init__(self, inf_distance : float = 512):
        super().__init__()
        self.inf_distance = inf_distance
        self.t = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def reset_parameters(self):
        self.t.data.random_()

    def forward(
        self,
        data : Batch,
    ) -> Batch:
        """ Add a diffusion distance matrix as an attribute to a batch """
        if not hasattr(data, "eigvecs") or not hasattr(data, "eigvals"):
            raise AttributeError(
                "Data has no attribute eigvals or eigvecs."
                " Data must first be preprocessed with the function add_laplacian_eigenpairs_to_data"
                " before LearnableDiffusionDistanceLayer can be used."
            )
        eigvecs_matrix = to_dense_adj(
            data.complete_graph_index,
            batch=data.batch,
            edge_attr=data.eigvecs
        )
        eigvals, real_nodes = to_dense_batch(
            data.eigvals,
            batch=data.batch
        )
        eigvals_diag = torch.diag_embed(torch.exp(-self.t * eigvals))
        eigvecs_matrix_T = eigvecs_matrix.permute((0, 2, 1))
        exp_laplacian = torch.bmm(torch.bmm(eigvecs_matrix, eigvals_diag), eigvecs_matrix_T)
        exp_laplacian_diagonal = torch.diagonal(exp_laplacian, dim1=1, dim2=2)
        diffusion_matrix = \
            exp_laplacian_diagonal.unsqueeze(1) \
            + exp_laplacian_diagonal.unsqueeze(2) \
            - 2*exp_laplacian
        component_mask = to_dense_adj(
            data.component_index, 
            batch=data.batch
        )
        neg_component_mask = torch.logical_not(component_mask)
        diffusion_matrix = component_mask*diffusion_matrix + self.inf_distance*neg_component_mask
        diffusion_matrix_flattened = from_dense_adj(diffusion_matrix, data.batch)
        if hasattr(data, "rpe"):
            data.rpe = torch.column_stack((
                data.rpe,
                diffusion_matrix_flattened
            )) 
        else:
            data.rpe = diffusion_matrix_flattened
        return data
    

class BiasEncoder(torch.nn.Module):
    def __init__(self, use_graph_token: bool = True):
        """ If using a graph token, appends an entry to the RPE.
        This encoder is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.

        Args:
            use_graph_token: If True, pads the attn_bias to account for the
            additional graph token that can be added by the ``NodeEncoder``.
        """
        super().__init__()
        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)

    def forward(self, data):
        """Computes the bias matrix that can be induced into multi-head attention
        via the attention mask.

        Adds the tensor ``attn_bias`` to the data object, optionally accounting
        for the graph token.
        """
        # To convert 2D matrices to dense-batch mode, one needs to decompose
        # them into index and value. One example is the adjacency matrix
        # but this generalizes actually to any 2D matrix
        bias = to_dense_adj(
            data.complete_graph_index, 
            batch=data.batch, 
            edge_attr=data.rpe
        )
        if bias.dim() == 3:
            # if we are only using a single distance,
            # add another dimension to the distance tensor
            bias = bias.unsqueeze(-1)
        if self.use_graph_token:
            # add extra entry on 2 and 3 entry
            bias = F.pad(bias, INSERT_GRAPH_TOKEN)
            # fill the extra entry with the learnable graph_token paramaeter
            bias[:, 1:, 0, :] = self.graph_token
            bias[:, 0, :, :] = self.graph_token
        data.attn_bias = bias
        return data


def add_graph_token(data, token):
    """Helper function to augment a batch of PyG graphs
    with a graph token each. Note that the token is
    automatically replicated to fit the batch.

    Args:
        data: A PyG data object holding a single graph
        token: A tensor containing the graph token values

    Returns:
        The augmented data object.
    """
    B = len(data.batch.unique())
    tokens = torch.repeat_interleave(token, B, 0)
    data.x = torch.cat([tokens, data.x], 0)
    data.batch = torch.cat(
        [torch.arange(0, B, device=data.x.device, dtype=torch.long), data.batch]
    )
    data.batch, sort_idx = torch.sort(data.batch)
    data.x = data.x[sort_idx]
    return data


class NodeEncoder(torch.nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_in_degree, 
        num_out_degree,     
        input_dropout = 0.0, 
        use_degree_embedding: bool = True,
        use_graph_token: bool = True
    ):
        """Implementation of the node encoder of Graphormer.
        This encoder is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.

        Args:
            embed_dim: The number of hidden dimensions of the model
            num_in_degree: Maximum size of in-degree to encode
            num_out_degree: Maximum size of out-degree to encode
            input_dropout: Dropout applied to the input features
            use_graph_token: If True, adds the graph token to the incoming batch.
        """
        super().__init__()

        self.use_degree_embedding = use_degree_embedding
        if self.use_degree_embedding:
            self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim)
            self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim)

        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))

        self.input_dropout = torch.nn.Dropout(input_dropout)
        
        self.reset_parameters()

    def forward(self, data):
        if self.use_degree_embedding:
            in_degree_encoding = self.in_degree_encoder(data.in_degrees)
            out_degree_encoding = self.out_degree_encoder(data.out_degrees)
            if hasattr(data, "x") and data.x.size(1) > 0:
                data.x = data.x + in_degree_encoding + out_degree_encoding
            else:
                data.x = in_degree_encoding + out_degree_encoding
        if self.use_graph_token:
            data = add_graph_token(data, self.graph_token)
        data.x = self.input_dropout(data.x)
        return data

    def reset_parameters(self):
        if self.use_degree_embedding:
            self.in_degree_encoder.weight.data.normal_(std=0.02)
            self.out_degree_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)
        

@register_node_encoder("RPETransformerBias")
class RPETransformerEncoder(torch.nn.Sequential):
    def __init__(self, dim_emb, *args, **kwargs):
        encoders = []
        if cfg.posenc_RPE.learnable_spectral_distance.enable:
            encoders.append( 
                LearnableSpectralDistanceLayer(
                    cfg.posenc_RPE.learnable_spectral_distance.hidden_dim,
                    cfg.posenc_RPE.inf_distance
                )
            )
        if cfg.posenc_RPE.learnable_diffusion_distance.enable:
            encoders.append(
                LearnableDiffusionDistanceLayer(
                    cfg.posenc_RPE.inf_distance
                )
            )
        if cfg.posenc_RPE.SPE.enable:
            encoders.append(
                SPELayer(
                    cfg.posenc_RPE.SPE.hidden_dim,
                    cfg.posenc_RPE.SPE.output_dim,
                    cfg.posenc_RPE.SPE.num_layers,
                    cfg.posenc_RPE.inf_distance,
                    cfg.posenc_RPE.SPE.as_distance
                )
            )
        
        encoders += [
            BiasEncoder(
                cfg.rpe_transformer.use_graph_token
            ),
            NodeEncoder(
                cfg.rpe_transformer.embed_dim,
                cfg.posenc_RPE.num_in_degrees,
                cfg.posenc_RPE.num_out_degrees,
                cfg.rpe_transformer.input_dropout,
                cfg.rpe_transformer.use_degree_embedding,
                cfg.rpe_transformer.use_graph_token
            ),
        ]
        if cfg.posenc_RPE.num_rpes == 0:  # No attn. bias encoder
            encoders = [encoders[-1]]
        super().__init__(*encoders)

if __name__=="__main__":
    import networkx as nx
    import torch_geometric

    G = nx.Graph()
    for i in range(3):
        G.add_edge(i, (i+1)%3)
    G.add_edge(3,4)
    G.add_edge(4,5)
    data = torch_geometric.utils.from_networkx(G)
    
    data = add_identity_to_data(data)
    rpe = to_dense_adj(data.complete_graph_index, edge_attr=data.rpe)
    print(rpe)

    
    
