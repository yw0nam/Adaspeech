import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
    
class ConditionalLayerNorm(nn.Module):
    """ Conditional LayerNorm """
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.scale_linear = nn.Linear(hidden_size, 256)
        self.bias_linear = nn.Linear(hidden_size, 256)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor ):
        # speaker_emb (B, max_src_len)
        
        max_mel_lens = x.shape[1]
        scale = self.scale_linear(speaker_emb).unsqueeze(1).expand(
            -1, max_mel_lens, -1
        )
        bias = self.bias_linear(speaker_emb).unsqueeze(1).expand(
            -1, max_mel_lens, -1
        )
        x = self.layer_norm(x)
        
        return scale * x + bias