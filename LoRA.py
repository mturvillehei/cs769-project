import numpy as np
import torch
import torch.nn as nn

class LoRA(nn.Module):

    def __init__(self, feat_in, feat_out, rank=8, alpha=1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A_mat = nn.Parameter(torch.empty(feat_in,rank))
        self.B_mat = nn.Parameter(torch.zeros(rank, feat_out))
        nn.init.kaiming_uniform_(self.A_mat, a=np.sqrt(5))

    def forward(self, x):
        return x.matmul(self.A_mat).matmul(self.B_mat)*(self.alpha/self.rank)



class LoRAAttentionLayer(nn.Module):
    def __init__(self, attn_layer, rank=8, alpha=1, trainable=False):
        super().__init__()
        self.base = attn_layer
        self.feat_in = self.feat_out = attn_layer.nx
        for param in self.base.parameters():
            param.requires_grad = trainable

        self.loraQ = LoRA(self.feat_in, self.feat_out, rank=rank, alpha=alpha)
        self.loraV = LoRA(self.feat_in, self.feat_out, rank=rank, alpha=alpha)
        self.loraQ.requires_grad = True
        self.loraV.requires_grad = True


    def forward(self,x):
        qkv = self.base(x)
        qkv = qkv.chunk(3, dim=-1)
        q,k,v = qkv[0],qkv[1],qkv[2]
        return torch.cat([q+self.loraQ(x), k, v + self.loraV(x)],dim=-1)


def add_lora_attention_layers(model, target_modules='c_attn', rank=8, alpha=1):
    to_change = []
    for name, module in model.named_modules():
       if target_modules in name:
           to_change.append((name,module))
    for i in to_change:
        setattr(model, i[0], LoRAAttentionLayer(i[1], rank=rank, alpha=alpha))


    return model