import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_networkx, get_laplacian

def set_intercomponent_distance_to_inf(
    data : Data,
    distance_matrix : torch.Tensor, 
    inf_distance : float
) -> torch.Tensor:
    """ Set the entries of `distance_matrix` corresponding to nodes in different components to `inf_distance` """
    component_mask = generate_component_mask(data).to(distance_matrix.device)
    neg_component_mask = torch.logical_not(component_mask)
    distance_matrix = component_mask*distance_matrix + inf_distance*neg_component_mask
    return distance_matrix

def get_complete_graph_index(n : int) -> torch.Tensor:
    """ Return the edge index of the complete graph on `n` nodes"""
    edge_index = torch.stack(
        torch.meshgrid(
            torch.arange(n),
            torch.arange(n)
        )
    ).flatten(1)
    return edge_index

def get_dense_laplacian_matrix(data : Data) -> torch.Tensor:
    """ Return the laplacian matrix of a graph """
    laplacian_edge_index, laplacian_edge_attr = get_laplacian(data.edge_index)
    laplacian_matrix = to_dense_adj(laplacian_edge_index, edge_attr=laplacian_edge_attr, max_num_nodes=data.num_nodes)
    return laplacian_matrix[0] 

def get_matrix_powers(
    matrix : torch.Tensor,
    num_powers : int
) -> torch.Tensor:
    """ Return a tensor containing the first `num_powers` powers of `matrix` """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square. input matrix has shape {matrix.shape}")
    powers_of_matrix = torch.Tensor(matrix.shape[0], matrix.shape[1], num_powers).to(matrix.device)
    powers_of_matrix[:,:,0] = matrix 
    for i in range(1, num_powers):
        powers_of_matrix[:,:,i] = powers_of_matrix[:,:,i-1] @ matrix 
    return powers_of_matrix 

def add_matrix_to_data(
    data : Data,
    matrix : torch.Tensor
) -> Data:
    """ Add the distance matrix as an attribute to this tensor """
    if hasattr(data, "rpe"):
        data.rpe = torch.column_stack((
            data.rpe,
            matrix.flatten(end_dim=1)
        )) 
    else:    
        data.rpe = matrix.flatten(end_dim=1)
    return data

def generate_component_mask(data : Data) -> torch.Tensor:
    """ Compute the nxn matrix where the i,j entry is 1 iff i and j are in the same connected component """
    graph = to_networkx(data, to_undirected=True)
    components = [list(component) for component in nx.connected_components(graph)]
    component_indicators = torch.zeros((len(graph), len(components)))
    for i, component in enumerate(components):
        component_indicators[component, i] = 1
    return component_indicators @ component_indicators.T

def add_component_mask_to_data(
    data : Data
) -> Data:
    """ Add the component mask as an edge index to a torch_geometric.data.Data object 
    
    To retrieve the component mask, use the command 
    ```
        component_mask = torch_geometric.utils.to_dense_adj(data.component_index)
    ```
    """   
    component_mask = generate_component_mask(data)
    data.component_index = torch.nonzero(component_mask).T
    return data


def add_degree_encoding(
    data : Data
) -> Data:
    graph : nx.DiGraph = to_networkx(data)

    data.in_degrees = torch.tensor([d for _, d in graph.in_degree()])
    data.out_degrees = torch.tensor([d for _, d in graph.out_degree()])

    max_in_degree = torch.max(data.in_degrees)
    max_out_degree = torch.max(data.out_degrees)
    if max_in_degree >= cfg.posenc_RPE.num_in_degrees:
        raise ValueError(
            f"Encountered in_degree: {max_in_degree}, set posenc_"
            f"RPETransformerBias.num_in_degrees to at least {max_in_degree + 1}"
        )
    if max_out_degree >= cfg.posenc_RPE.num_out_degrees:
        raise ValueError(
            f"Encountered out_degree: {max_out_degree}, set posenc_"
            f"RPETransformerBias.num_out_degrees to at least {max_out_degree + 1}"
        )
    
    return data


def add_edge_features_to_data(
    data : Data 
) -> Data:
    """ Add the edge features of a graph as an rpe """
    if not hasattr(data, "edge_attr"):
        raise ValueError("Input data has no attribute edge_attr")
    feature_matrix = to_dense_adj(
        data.edge_index,
        edge_attr=data.edge_attr,
        max_num_nodes=data.num_nodes
    )[0]
    data = add_matrix_to_data(data, feature_matrix)
    return data


def add_laplacian_eigenpairs_to_data(
    data : Data,
    device : str
) -> Data:
    """ Add `eigvals` and `eigvecs` of the unnormalized Laplacian to a torch_geometric.data.Data object

    Add torch.Tensor containing the the eigvals and eigvecs to a data object. 
    eigvecs containing the eigenvectors in a flattened form. 
    To convert them to a matrix, use the command
    ```
    eigvecs_matrix = torch_geometric.utils.to_dense_adj(
        data.complete_graph_index,
        edge_attr=data.eigvecs
    )
    ```
    To recover the Laplacian, use the command
    ```
    eigvecs_matrix = torch_geometric.utils.to_dense_adj(
        data.complete_graph_index,
        edge_attr=data.eigvecs
    )
    eigvals_diag = torch.diag_embed(data.eigvals)
    laplacian = eigvecs_matrix @ eigvecs_diag @ eigvec_matrix.T
    ```
    """
    laplacian = get_dense_laplacian_matrix(data).to(device)
    eigvals, eigvecs = torch.linalg.eigh(laplacian)
    data.eigvals = eigvals.cpu() 
    data.eigvecs = eigvecs.cpu().flatten()
    data = add_component_mask_to_data(data)
    return data

def add_adjacency_to_data(
    data : Data
) -> Data:
    """ Add the adjacency matrix as a distance matrix to a torch_geometric.data.Data object """
    adjacency_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    data = add_matrix_to_data(data, adjacency_matrix)
    return data


def add_resistance_to_data(
     data : Data,
     inf_distance : float,
     device : str
 ) -> Data:
    """ Compute all-pairs effective resistance and add it as an attribute to a torch_geometric.data.Data object

    This method adds the attributes `complete_graph_index` and `resistances` to a torch_geomtric.data.Data object.
    `complete_graph_index` is the edge index of the complete graph, e.g [[0,0], [0,1], ...]
    `distances` is the list of all-pairs effective resistances in the graph. 
    A 2d resistance matrix can be reconstructed using the command
        `resistance_matrix = torch.utils.to_dense_adj(data.complete_graph_index, edge_attr=data.rpe)`

    Args:
        data (torch_geomtric.data.Data) : a torch_geometric graph
        inf_distance (float) : the value used to represent the resistance
        between nodes in different connected components

    """
    laplacian = get_dense_laplacian_matrix(data).to(device)
    pinv = torch.linalg.pinv(laplacian, hermitian=True)
    pinv_diagonal = torch.diagonal(pinv)
    resistance_matrix = pinv_diagonal.unsqueeze(0) + pinv_diagonal.unsqueeze(1) - 2*pinv
    resistance_matrix = set_intercomponent_distance_to_inf(data, resistance_matrix, inf_distance)
    data = add_matrix_to_data(data, resistance_matrix.cpu())
    return data

def add_diffusion_to_data(
    data : Data,
    t : float,
    inf_distance : float,
    device : str
) -> Data:
    laplacian = get_dense_laplacian_matrix(data).to(device)
    exp_laplacian = torch.matrix_exp(-t*laplacian)
    exp_laplacian_diagonal = torch.diag(exp_laplacian)
    diffusion_matrix = \
        exp_laplacian_diagonal.unsqueeze(0) \
        + exp_laplacian_diagonal.unsqueeze(1) \
        - 2*exp_laplacian
    diffusion_matrix = set_intercomponent_distance_to_inf(data, diffusion_matrix, inf_distance)
    data = add_matrix_to_data(data, diffusion_matrix.cpu())
    return data

def add_constant_rpe_to_data(
    data : Data, 
    inf_distance : float
) -> Data:
    component_mask = generate_component_mask(data)
    neg_component_mask = torch.logical_not(component_mask)
    distance_matrix = inf_distance*neg_component_mask
    data = add_matrix_to_data(data, distance_matrix) 
    return data

def add_shortest_path_to_data(
    data : Data, 
    inf_distance : float
) -> Data:
    graph = to_networkx(data, to_undirected=True)
    distance_matrix = torch.from_numpy(nx.floyd_warshall_numpy(graph))
    distance_matrix = torch.nan_to_num(distance_matrix, posinf=inf_distance)
    data = add_matrix_to_data(data, distance_matrix) 
    return data

def add_laplacian_to_data(
    data : Data
) -> Data: 
    """ Add laplacian as rpe to data  """
    data = add_matrix_to_data(data, get_dense_laplacian_matrix(data))
    return data

def add_pseudoinverse_to_data(
    data : Data,
    device : str
) -> Data:
    """ Add pseudoinverse as rpe to data """
    laplacian = get_dense_laplacian_matrix(data).to(device)
    laplacian_pinv = torch.linalg.pinv(laplacian, hermitian=True)
    data = add_matrix_to_data(data, laplacian_pinv.cpu())
    return data

def add_identity_to_data(
    data : Data,
) -> Data:
    """ Add the identity matrix as rpe to data"""
    identity = torch.eye(data.num_nodes)
    data = add_matrix_to_data(data, identity)
    return data

def add_powers_of_laplacian_to_data(
    data : Data,
    num_powers : int
) -> Data:
    """ Add the first `num_powers` of the Laplacian to data """
    laplacian = get_dense_laplacian_matrix(data)
    powers_of_laplacian = get_matrix_powers(laplacian, num_powers)
    data = add_matrix_to_data(data, powers_of_laplacian)
    return data

def add_powers_of_pseudoinverse_to_data(
    data : Data,
    num_powers : int,
    device : str
) -> Data:
    """ Add powers of the pseudoinverse as rpe to data """
    laplacian = get_dense_laplacian_matrix(data).to(device)
    pseudoinverse = torch.linalg.pinv(laplacian, hermitian=True)
    powers_of_pseudoinverse = get_matrix_powers(pseudoinverse, num_powers)
    data = add_matrix_to_data(data, powers_of_pseudoinverse.cpu())
    return data

def add_powers_of_adjacency_to_data(
    data : Data,
    num_powers : int
) -> Data:
    adjacency_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    powers_of_adjacency = get_matrix_powers(adjacency_matrix, num_powers)
    data = add_matrix_to_data(data, powers_of_adjacency)
    return data


def add_rpe_to_data(
    data : Data, 
    cfg 
) -> Data:
    """ Add RPEs specified by `cfg` to the data """    
    if cfg.posenc_RPE.resistance_distance:
        data = add_resistance_to_data(
            data, 
            cfg.posenc_RPE.inf_distance,
            device=cfg.accelerator
        )
    if cfg.posenc_RPE.adjacency:
        data = add_adjacency_to_data(
            data
        )
    if cfg.posenc_RPE.shortest_path_distance:
        data = add_shortest_path_to_data(
            data,
            cfg.posenc_RPE.inf_distance
        )
    if cfg.posenc_RPE.diffusion_distance:
        data = add_diffusion_to_data(
            data,
            cfg.posenc_RPE.diffusion_coefficient, 
            cfg.posenc_RPE.inf_distance,
            device=cfg.accelerator
        )
    if cfg.posenc_RPE.laplacian:
        data = add_laplacian_to_data(
            data 
        )
    if cfg.posenc_RPE.pseudoinverse:
        data = add_pseudoinverse_to_data(
            data,
            device=cfg.accelerator
        )
    if cfg.posenc_RPE.powers_of_pseudoinverse:
        data = add_powers_of_pseudoinverse_to_data(
            data,
            cfg.posenc_RPE.powers_of_pseudoinverse,
            device=cfg.accelerator
        )
    if cfg.posenc_RPE.powers_of_adjacency:
        data = add_powers_of_adjacency_to_data(
            data, 
            cfg.posenc_RPE.powers_of_adjacency
        )
    if cfg.posenc_RPE.powers_of_laplacian:
        data = add_powers_of_laplacian_to_data(
            data, 
            cfg.posenc_RPE.powers_of_laplacian
        )
    if cfg.posenc_RPE.edge_features:
        data = add_edge_features_to_data(
            data
        )
    if cfg.posenc_RPE.identity_distance:
        data = add_identity_to_data(
            data
        )
    if cfg.posenc_RPE.learnable_diffusion_distance.enable \
        or cfg.posenc_RPE.learnable_spectral_distance.enable \
        or cfg.posenc_RPE.SPE.enable:
        data = add_laplacian_eigenpairs_to_data(
            data,
            device=cfg.accelerator
        )
    # add the complete graph index to the data so we can reconstruct the RPE matrix later
    data.complete_graph_index = get_complete_graph_index(data.num_nodes)

    return data