from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('cfg_ape_deepsets')
def set_cfg_gt(cfg):
    cfg.deepsets_ape_encoder = CN()
    cfg.deepsets_ape_encoder.enable = False 
    cfg.deepsets_ape_encoder.input_dim = 128 
    cfg.deepsets_ape_encoder.hidden_dim = 128 
    cfg.deepsets_ape_encoder.output_dim = 1 
    cfg.deepsets_ape_encoder.num_hidden_layers = 2
    cfg.deepsets_ape_encoder.ape_type = None