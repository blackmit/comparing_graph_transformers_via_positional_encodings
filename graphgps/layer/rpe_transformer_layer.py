import torch
from graphgps.layer.multiheaded_attention_layer import MultiHeadAttentionLayer
from graphgps.layer.rpe_encoder_layer import RPEEncoderLayer
from torch_geometric.utils import to_dense_batch
from typing import Optional

class RPETransformerLayer(torch.nn.Module):
    def __init__(
        self, 
        embed_dim : int,
        distance_encoder_hidden_dim: int,
        num_heads: int, 
        num_kernels: int, 
        num_rpes: int, 
        dropout: float,
        attention_dropout: float, 
        mlp_dropout: float,
        use_gaussians: bool,
        use_add_bias: bool,
        use_mult_bias: bool
    ):
        """ Implementation of the RPE Transformer layer.

        This layer is based on the implementation of the Graphormer at:
            https://github.com/microsoft/Graphormer/tree/v1.0
        as well as the implementation of the RPE Transformer layer at:
            https://github.com/lsj2408/Graphormer-GD

        Args:
            embed_dim: The number of hidden dimensions of the model
            distance_encoder_hidden_dim: The number of hidden dimensions of the distance encoder
            num_heads: The number of heads of the transformer model
            num_kernels : The number of Gaussian kernels for the distance encoder
            num_rpes : The number of different RPEs 
            dropout: Dropout applied after the attention and after the MLP
            attention_dropout: Dropout applied within the attention
            mlp_dropout: Dropout applied within the MLP
            use_gaussians: Whether or not to use Gaussians in the distance encoder
            use_add_bias: Whether or not to use additive bias 
            use_mult_bias: Whether or not to use mulitiplicative bias 
        """
        super().__init__()
        self.attention = MultiHeadAttentionLayer(
            embed_dim,
            num_heads,
            attention_dropout,
        )
        if use_add_bias:
            self.add_bias_encoder = RPEEncoderLayer(
                num_kernels=num_kernels,
                num_heads=num_heads,
                num_rpes=num_rpes,
                hidden_dim=distance_encoder_hidden_dim,
                bias_type="add",
                use_gaussians=use_gaussians
            )
        if use_mult_bias:
            self.mult_bias_encoder = RPEEncoderLayer(
                num_kernels=num_kernels,
                num_heads=num_heads,
                num_rpes=num_rpes,
                hidden_dim=distance_encoder_hidden_dim,
                bias_type="mult",
                use_gaussians=use_gaussians
            )
        self.use_add_bias = use_add_bias
        self.use_mult_bias = use_mult_bias
            
        self.input_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

        # We follow the GDGraphormer paper in that all hidden dims are
        # equal to the embedding dim
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, data):
        if self.use_mult_bias:
            data = self.mult_bias_encoder(data) 
        if self.use_add_bias:
            data = self.add_bias_encoder(data)
        X = data.x
        X = self.input_norm(X)
        X, real_nodes = to_dense_batch(X, data.batch)
        X = self.attention(
            X, ~real_nodes,
            attn_mult=data.mult_bias if hasattr(data, "mult_bias") else None,
            attn_add=data.add_bias if hasattr(data, "add_bias") else None
        )[real_nodes]
        X = self.dropout(X) + data.x
        data.x = self.mlp(X) + X
        return data


