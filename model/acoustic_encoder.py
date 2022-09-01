import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from model.modules import Conv

class UtteranceEncoder(nn.Module):

    def __init__(self, in_dim: int,
                n_layers: int = 2,
                hidden_size: int = 256,
                kernel_size: int = 5,
                dropout_rate: float = 0.5,
                stride: int = 3):
        super(UtteranceEncoder, self).__init__()
        self.conv = nn.ModuleList()
        for idx in range(n_layers):
            layer_hs = in_dim if idx == 0 else hidden_size
            self.conv += [
                nn.Sequential(
                    Conv(
                        layer_hs,
                        hidden_size,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout_rate),
                )
            ]

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for layer in self.conv:
            x = layer(x)  # (B, Ts, C)
        
        # NOTE: calculate in log domain
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, x.size(2))  # (B, C, 1)

        return x


class PhonemeLevelEncoder(nn.Module):

    def __init__(self, in_dim: int,
                n_layers: int = 2,
                hidden_size: int = 256,
                latent_hs: int = 4,
                kernel_size: int = 3,
                dropout_rate: float = 0.5,
                stride: int = 1):
        super(PhonemeLevelEncoder, self).__init__()
        self.conv = nn.ModuleList()
        for idx in range(n_layers):
            layer_hs = in_dim if idx == 0 else hidden_size
            self.conv += [
                nn.Sequential(
                    Conv(
                        layer_hs,
                        hidden_size,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout_rate),
                )
            ]

        self.latent_linear = nn.Linear(hidden_size, latent_hs)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            x = f(x)  # (B, Lmax, C)
            
        x = self.latent_linear(x)  # (B, Lmax, 4)
        return x


class PhonemeLevelPredictor(nn.Module):

    def __init__(self, in_dim: int,
                n_layers: int = 2,
                hidden_size: int = 256,
                latent_hs: int = 4,
                kernel_size: int = 3,
                dropout_rate: float = 0.5,
                stride: int = 1):
        super(PhonemeLevelPredictor, self).__init__()
        self.conv = nn.ModuleList()
        for idx in range(n_layers):
            layer_hs = in_dim if idx == 0 else hidden_size
            self.conv += [
                nn.Sequential(
                    Conv(
                        layer_hs,
                        hidden_size,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout_rate),
                )
            ]

        self.latent_linear = nn.Linear(hidden_size, latent_hs)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            x = f(x)  # (B, Lmax, C)

        x = self.latent_linear(x)  # (B, Lmax, 4)
        return x
