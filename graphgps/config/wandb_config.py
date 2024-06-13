from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_wandb')
def set_cfg_wandb(cfg):
    """Weights & Biases tracker configuration.
    """

    # WandB group
    cfg.wandb = CN()

    # Use wandb or not
    cfg.wandb.use = False

    # Wandb entity name, should exist beforehand
    cfg.wandb.entity = "gtransformers"

    # Wandb project name, will be created in your team if doesn't exist already
    cfg.wandb.project = "gtblueprint"

    # Optional run name
    cfg.wandb.name = ""

    # Options for hyperparameter sweep
    cfg.wandb.sweep = CN()
    cfg.wandb.sweep.enable = False
    cfg.wandb.sweep.id = None 
    cfg.wandb.sweep.count = 1

