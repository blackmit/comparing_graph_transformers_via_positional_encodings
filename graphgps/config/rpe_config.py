from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_rpe')
def set_cfg_gt(cfg):
    cfg.posenc_RPE = CN()
    cfg.posenc_RPE.dim_pe = 0
    cfg.posenc_RPE.enable = False
    cfg.posenc_RPE.inf_distance = 512.0
    cfg.posenc_RPE.num_in_degrees = None
    cfg.posenc_RPE.num_out_degrees = None
    # Types of RPE
    cfg.posenc_RPE.adjacency = False
    cfg.posenc_RPE.shortest_path_distance = False
    cfg.posenc_RPE.resistance_distance = False 
    cfg.posenc_RPE.diffusion_distance = False 
    cfg.posenc_RPE.diffusion_coefficient = 0.0
    cfg.posenc_RPE.laplacian = False
    cfg.posenc_RPE.pseudoinverse = False
    cfg.posenc_RPE.identity_distance = False
    cfg.posenc_RPE.edge_features = False
    cfg.posenc_RPE.powers_of_pseudoinverse = 0
    cfg.posenc_RPE.powers_of_laplacian = 0
    cfg.posenc_RPE.powers_of_adjacency = 0  
    cfg.posenc_RPE.num_rpes = 0 
    # Learnable distances
    ldd = cfg.posenc_RPE.learnable_diffusion_distance = CN()
    ldd.enable = False
    lsd = cfg.posenc_RPE.learnable_spectral_distance = CN()
    lsd.enable = False
    lsd.hidden_dim = 32
    spe = cfg.posenc_RPE.SPE = CN()
    spe.enable = False 
    spe.hidden_dim = 32
    spe.output_dim = 1
    spe.num_layers = 2
    spe.as_distance = False


