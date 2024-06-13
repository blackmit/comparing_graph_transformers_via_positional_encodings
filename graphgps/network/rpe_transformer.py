import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.rpe_transformer_layer import RPETransformerLayer
from graphgps.layer.rpe_encoder_layer import RPEEncoderLayer


@register_network('RPETransformer')
class RPETransformerModel(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        layers = []
        if not cfg.rpe_transformer.update_bias_each_layer:
            if cfg.rpe_transformer.use_add_bias:
                layers.append(RPEEncoderLayer(
                    num_kernels=cfg.rpe_transformer.num_kernels,
                    num_heads=cfg.rpe_transformer.num_heads,
                    num_rpes=cfg.posenc_RPE.num_rpes,
                    hidden_dim=cfg.rpe_transformer.distance_encoder_hidden_dim,
                    bias_type="add",
                    use_gaussians=cfg.rpe_transformer.use_gaussians
                ))
            if cfg.rpe_transformer.use_mult_bias:
                layers.append(RPEEncoderLayer(
                    num_kernels=cfg.rpe_transformer.num_kernels,
                    num_heads=cfg.rpe_transformer.num_heads,
                    num_rpes=cfg.posenc_RPE.num_rpes,
                    hidden_dim=cfg.rpe_transformer.distance_encoder_hidden_dim,
                    bias_type="mult",
                    use_gaussians=cfg.rpe_transformer.use_gaussians
                ))
        for _ in range(cfg.rpe_transformer.num_layers):
            layers.append(RPETransformerLayer(
                embed_dim=cfg.rpe_transformer.embed_dim,
                distance_encoder_hidden_dim=cfg.rpe_transformer.distance_encoder_hidden_dim,
                num_kernels=cfg.rpe_transformer.num_kernels,
                num_heads=cfg.rpe_transformer.num_heads,
                num_rpes=cfg.posenc_RPE.num_rpes,
                dropout=cfg.rpe_transformer.dropout,
                attention_dropout=cfg.rpe_transformer.attention_dropout,
                mlp_dropout=cfg.rpe_transformer.mlp_dropout,
                use_gaussians=cfg.rpe_transformer.use_gaussians,
                use_add_bias=(cfg.rpe_transformer.update_bias_each_layer and cfg.rpe_transformer.use_add_bias),
                use_mult_bias=(cfg.rpe_transformer.update_bias_each_layer and cfg.rpe_transformer.use_mult_bias)
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
