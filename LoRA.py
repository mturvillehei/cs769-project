import numpy as np
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention, GPT2FlashAttention2

class LoRA(nn.Module):

    def __init__(self, feat_in, feat_out, rank=8, alpha=16):
        super().__init__()
        self.scale = torch.tensor((alpha/rank), dtype=torch.bfloat16)
        self.A_mat = nn.Parameter(torch.empty((feat_in,rank), dtype=torch.bfloat16))
        self.B_mat = nn.Parameter(torch.zeros((rank, feat_out), dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.A_mat, a=np.sqrt(5))

    def forward(self, x):
        return torch.mul(torch.matmul(x,torch.matmul(self.A_mat,self.B_mat)), self.scale)



class LoRAAttentionLayer(nn.Module):
    def __init__(self, attn_layer:nn.Module, rank=8, alpha=16, trainable=False):
        super().__init__()
        self.base = attn_layer
        self.feat_in = self.feat_out = attn_layer.nx
        for param in self.base.parameters():
            param.requires_grad = trainable

        self.loraQ = LoRA(self.feat_in, self.feat_out, rank=rank, alpha=alpha)
        self.loraV = LoRA(self.feat_in, self.feat_out, rank=rank, alpha=alpha)


    def forward(self,x):
        qkv = self.base(x)
        qkv = qkv.chunk(3, dim=-1)
        return torch.cat([qkv[0]+self.loraQ(x), qkv[1], qkv[2] + self.loraV(x)],dim=-1)


def add_lora_attention_layers(model, target_modules='c_attn', rank=8, alpha=16):
    to_change = []
    for name, module in model.named_modules():
       if type(module) is GPT2SdpaAttention or type(module) is GPT2FlashAttention2:
           to_change.append((name,module))
    for i in to_change:
        i[1].set_submodule(target_modules, LoRAAttentionLayer(i[1].c_attn, rank=rank, alpha=alpha))
    for n,p in model.named_parameters():
        if 'lora' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    return model