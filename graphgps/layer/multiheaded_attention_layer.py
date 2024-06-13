import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_bias=True):
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout = nn.Dropout(dropout)
        
        self.Q = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.K = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.V = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.fc_out = nn.Linear(embed_dim, embed_dim, bias=use_bias)  # Final output layer

    def forward(self, X, real_nodes=None, attn_mult=None, attn_add=None):
        batch_size = X.size(0)
        
        Q_h = self.Q(X)
        K_h = self.K(X)
        V_h = self.V(X)
        
        Q_h = Q_h.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_h = K_h.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_h = V_h.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention calculation
        score = torch.matmul(Q_h, K_h.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        if real_nodes is not None:
            real_nodes = real_nodes.unsqueeze(1).unsqueeze(-1)
            score = score * real_nodes

        if attn_add is not None:
            score = score + attn_add
        
        score = F.softmax(score, dim=-1)
        
        if attn_mult is not None:
            score = score * attn_mult
        
        score = self.dropout(score)
        
        # Output calculation
        output = torch.matmul(score, V_h)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.fc_out(output)  # Final linear transformation
        
        return output

# External class for generating learnable attn_add tensor
class LearnableAdd(nn.Module):
    def __init__(self, shape):
        super(LearnableAdd, self).__init__()
        self.add_tensor = nn.Parameter(torch.rand(shape))

    def forward(self):
        return self.add_tensor

# External class for generating learnable attn_mult tensor
class LearnableMul(nn.Module):
    def __init__(self, shape):
        super(LearnableMul, self).__init__()
        self.mul_tensor = nn.Parameter(torch.rand(shape))

    def forward(self):
        return self.mul_tensor

if __name__=="__main__":    
    # Instantiate the modules
    embed_dim = 64
    num_heads = 4
    batch_size = 32
    seq_len = 10

    mha = MultiHeadAttentionLayer(embed_dim, num_heads)
    learnable_add = LearnableAdd((batch_size, num_heads, seq_len, seq_len))
    learnable_mul = LearnableMul((batch_size, num_heads, seq_len, seq_len))

    # Generate some random data
    X = torch.rand((batch_size, seq_len, embed_dim))
    real_nodes = torch.ones((batch_size, seq_len, seq_len))

    # Forward pass
    attn_add = learnable_add()
    attn_mult = learnable_mul()

    # Initialize the optimizer and loss function
    optimizer = optim.SGD(list(mha.parameters()) + list(learnable_add.parameters()) + list(learnable_mul.parameters()), lr=0.01)
    loss_fn = nn.MSELoss()

    # Perform multiple steps of backpropagation
    for step in range(300):
        optimizer.zero_grad()
        output = mha(X, real_nodes=real_nodes, attn_mult=attn_mult, attn_add=attn_add)
        target = torch.rand_like(output)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1} - Loss: {loss.item()}")
