import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

import numpy as np



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., dy_dim = 16):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.pos_linear = nn.Sequential(
            nn.Linear(3, dy_dim, bias=False),
            nn.GELU(),
            nn.Linear(dy_dim, 3, bias=False),
        )
        self.neg_linear = nn.Sequential(
            nn.Linear(3, dy_dim, bias=False),
            nn.GELU(),
            nn.Linear(dy_dim, 3, bias=False),
        )
        self.cross_linear = nn.Linear(3, 3, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        

    def forward(self, xyzs, features):
        b, n, _, h = *features.shape, self.heads
        
        norm_features = self.norm(features)
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)                             # [b, h, m, d]


        # xyzs_flatten = xyzs                                                     # [b, m, 3]

        delta_xyzs = torch.unsqueeze(input=xyzs, dim=1) - torch.unsqueeze(input=xyzs, dim=2)        # [b, m, m, 3]


        pos_output = self.pos_linear(F.relu(delta_xyzs)) # [b, m, m, 3]
        neg_output = self.neg_linear(F.relu(-delta_xyzs))   # [b, m, m, 3]
        
        pos_delta = self.leaky_relu(pos_output)
        neg_delta = self.leaky_relu(neg_output)

        
        cross_delta = self.leaky_relu(self.cross_linear(pos_delta*neg_delta))                                 # [b, m, m, 3]

        delta_xyzs =  cross_delta * delta_xyzs                                                            # [b, m, m, 3]
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                                             # [b, h, m, m]
        attn = dots.softmax(dim=-1)

        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                                          # [b, h, m, d]
        
        attn = torch.unsqueeze(input=attn, dim=4)                                                                   # [b, h, m, m, 1]
        delta_xyzs = torch.unsqueeze(input=delta_xyzs, dim=1)                                                       # [b, 1, m, m, 3]
        delta_xyzs = torch.sum(input=attn*delta_xyzs, dim=3, keepdim=False)                                         # [b, h, m, 3]

        displacement_features = self.spatial_op(delta_xyzs)                                                         # [b, h, m, d]

        out = v + displacement_features
        out = rearrange(out, 'b h m d -> b m (h d)')
        out =  self.to_out(out)
        
        
        return out + features

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        dy_dim = 16

    def forward(self, xyzs, features):

        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features
    





class tAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., n_frame=6):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self,features):
        b, n, _, h = *features.shape, self.heads # b: batch_size, l: time_step, n: num_points, h: num_heads

        norm_features = self.norm(features)

        qkv = self.to_qkv(norm_features).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)                             # [b, h, l, d]

        dots=einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                                             # [b, h, l, l]
        attn = dots.softmax(dim=-1)

        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                                          # [b, h, l, d]
        
        out = rearrange(v, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out + features



class temperal_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                tAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, features):
        for attn, ff in self.layers:
            features = attn(features)
            features = ff(features)
        return features


    
if __name__ == '__main__':
    model = Transformer(dim=80, depth=4, heads=8, dim_head=64, mlp_dim=80, dropout=0.1)
    xyzs = torch.rand(1, 6*12, 3)
    features = torch.rand(1, 6*12, 80)
    
    output = model(xyzs, features)
    
    print(output.shape)
    
