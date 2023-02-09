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
    
# class ConditionalLayerNorm(nn.Module):
#     """ Conditional LayerNorm """
#     def __init__(self, hidden_size: int = 256):
#         super().__init__()
#         self.scale_linear = nn.Linear(hidden_size, hidden_size)
#         self.bias_linear = nn.Linear(hidden_size, hidden_size)
#         self.layer_norm = nn.LayerNorm(hidden_size)
        
#     def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor ):
#         # speaker_emb (B, hs)
#         # x (B, max_mel_lens, hs)
#         max_mel_lens = x.shape[1]
#         scale = self.scale_linear(speaker_emb).unsqueeze(1).expand(
#             -1, max_mel_lens, -1
#         )
#         bias = self.bias_linear(speaker_emb).unsqueeze(1).expand(
#             -1, max_mel_lens, -1
#         )
#         x = self.layer_norm(x)
        
#         return scale * x + bias

class ConditionalLayerNorm(nn.Module):
    """ Conditional LayerNorm """
    def __init__(self, hidden_size: int = 256, speaker_embedding_dim: int = 256, epsilon=1e-5):
        super(ConditionalLayerNorm, self).__init__()
        
        self.epsilon = epsilon
        self.W_scale = nn.Linear(speaker_embedding_dim, hidden_size)
        self.W_bias = nn.Linear(speaker_embedding_dim, hidden_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)
        
    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor ):
        # speaker_emb (B, hs)
        # x (B, max_mel_lens, hs)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        x_out = (x - mean) / std
        scale = self.W_scale(speaker_emb)
        bias = self.W_bias(speaker_emb)
        x_out *= scale.unsqueeze(1)
        x_out += bias.unsqueeze(1)
        
        return x_out