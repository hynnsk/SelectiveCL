import math
import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
from torch import randperm as perm



class TRDecoder(nn.Module):

    def __init__(self, dim=384, reduced_dim=384, hidden_dim=2048, nhead=1, dropout=0.1):
        super().__init__()
        # reduced dim 90
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.f1 = nn.Conv2d(dim, reduced_dim, (1, 1))
        self.f2 = nn.Sequential(nn.Conv2d(dim, dim, (1, 1)),
                                  nn.ReLU(),
                                  nn.Conv2d(dim, reduced_dim, (1, 1)))

        num_queries = 224**2 // 16**2
        self.query_pos = nn.Parameter(torch.randn(num_queries, dim), requires_grad=True) # 196,384

    def forward(self, tgt, drop=nn.Identity()):
        tgt = tgt.transpose(0, 1) # 196,8,384
        q = k = tgt + self.query_pos.unsqueeze(1) # 196,8,384
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + self.query_pos.unsqueeze(1), key=tgt + self.query_pos.unsqueeze(1), value=tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        tgt = transform(tgt.transpose(0, 1))
        tgt = self.f1(drop(tgt)) + self.f2(drop(tgt))
        tgt = untransform(tgt)
        return tgt



def transform(x):
    """
    B, P, D => B, D, root(P), root(P)

    Ex) 128, 400, 768 => 128, 768, 20, 20
    """
    B, P, D = x.shape
    return x.permute(0, 2, 1).view(B, D, int(math.sqrt(P)), int(math.sqrt(P)))

def untransform(x):
    """
    B, D, P, P => B, P*P, D,

    Ex) 128, 768, 20, 20 => 128, 400, 768
    """
    B, D, P, P = x.shape
    return x.view(B, D, -1).permute(0, 2, 1)