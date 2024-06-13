from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('cfg_ign_encoder')
def set_cfg_gt(cfg):
    cfg.ign_encoder = CN()
    cfg.ign_encoder.enable = False 
    cfg.ign_encoder.num_hidden_layers = 2 
    cfg.ign_encoder.hidden_dim = 128 
    cfg.ign_encoder.out_dim = 64 # this must match cfg.rpe_transformer.embed_dim if type == sum
                                              # or (cfg.rpe_transformer.embed_dim - {node_feature}.dim) if type == concatenation 
    cfg.ign_encoder.type = "sum" # "sum" or "concatenation"
