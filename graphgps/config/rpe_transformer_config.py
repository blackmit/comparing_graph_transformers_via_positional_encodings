from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_rpe_transformer')
def set_cfg_gt(cfg):
    cfg.rpe_transformer = CN()
    cfg.rpe_transformer.num_layers = 6
    cfg.rpe_transformer.embed_dim = 80
    cfg.rpe_transformer.distance_encoder_hidden_dim = None
    cfg.rpe_transformer.num_heads = 4
    cfg.rpe_transformer.num_kernels = 128
    cfg.rpe_transformer.dropout = 0.0
    cfg.rpe_transformer.attention_dropout = 0.0
    cfg.rpe_transformer.mlp_dropout = 0.0
    cfg.rpe_transformer.input_dropout = 0.0
    cfg.rpe_transformer.use_graph_token = True
    cfg.rpe_transformer.use_degree_embedding = True
    cfg.rpe_transformer.update_bias_each_layer = True
    cfg.rpe_transformer.use_gaussians = True
    cfg.rpe_transformer.use_add_bias = True 
    cfg.rpe_transformer.use_mult_bias = True 

