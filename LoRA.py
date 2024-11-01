import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA(nn.Module):

    def __init__(self, feat_in, feat_out, rank=8, alpha=32):
        super().__init__()
        self.scale = torch.tensor((alpha/rank), dtype=torch.bfloat16)
        self.A_mat = nn.Linear(feat_in, rank, bias=False)
        self.B_mat = nn.Linear(rank, feat_out, bias=False)
        nn.init.kaiming_uniform_(self.A_mat.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B_mat.weight)

    def forward(self, x):
        return self.B_mat(self.A_mat(x)*self.scale)



class LoRAAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output embedding
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.loraQ = LoRA(config.n_embd, config.n_embd, rank=config.lora_r, alpha=config.lora_a)
        self.loraV = LoRA(config.n_embd, config.n_embd, rank=config.lora_r, alpha=config.lora_a)
        self.n_head = config.n_head
        self.n_embd = config.n_embd


    def forward(self,x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q + self.loraQ(q)
        v = v + self.loraV(v)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        return self.c_proj(y)
