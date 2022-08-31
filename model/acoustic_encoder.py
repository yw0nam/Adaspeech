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
        
        
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        
        # NOTE: calculate in log domain
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, x.size(2))  # (B, C, 1)

        return x


class PhonemeLevelEncoder(nn.Module):

    def __init__(self, idim: int,
                    n_layers: int = 2,
                    n_chans: int = 256,
                    out: int = 4,
                    kernel_size: int = 3,
                    dropout_rate: float = 0.5,
                    stride: int = 1):
        super(PhonemeLevelEncoder, self).__init__()
        self.conv = nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                nn.Sequential(
                    Conv(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(n_chans),
                    nn.Dropout(dropout_rate),
                )
            ]

        self.linear = nn.Linear(n_chans, out)

    def forward(self,
                xs: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            xs = f(xs)  # (B, C, Lmax)


        xs = self.linear(xs.transpose(1, 2))  # (B, Lmax, 4)

        return xs


class PhonemeLevelPredictor(nn.Module):

    def __init__(self, idim: int,
                 n_layers: int = 2,
                 n_chans: int = 256,
                 out: int = 4,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.5,
                 stride: int = 1):
        super(PhonemeLevelPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

        self.linear = torch.nn.Linear(n_chans, out)

    def forward(self,
                xs: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax)

        return xs