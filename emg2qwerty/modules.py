# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from collections.abc import Sequence

import torch
from torch import nn
import math
from dataclasses import dataclass
from typing import Optional

class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)



class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        x = x + self.pe[:T].unsqueeze(1)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Input:  (T, N, D)
    Output: (T, N, D)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_sinusoidal_pos_emb: bool = True,
        max_len: int = 200000,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self._max_len = max_len
        self._use_sinusoidal = use_sinusoidal_pos_emb

        if use_sinusoidal_pos_emb:
            self.pos_emb = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout,
            )
        else:
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.pos_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def _make_padding_mask(self, lengths: torch.Tensor, T: int) -> torch.Tensor:
        return torch.arange(T, device=lengths.device)[None, :] >= lengths[:, None]

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError(f"Expected inputs of shape (T, N, D), got {tuple(inputs.shape)}")

        T, N, D = inputs.shape
        if D != self.d_model:
            raise ValueError(f"Expected D == d_model ({self.d_model}), got D={D}")
        if T > self._max_len:
            raise ValueError(f"T={T} exceeds max_len={self._max_len}. Increase max_len in TransformerEncoder.")

        x = inputs

        if self._use_sinusoidal:
            x = self.pos_emb(x)
        else:
            pos = torch.arange(T, device=x.device)
            x = x + self.pos_emb(pos).unsqueeze(1)
            x = self.pos_dropout(x)

        src_key_padding_mask = (
            self._make_padding_mask(lengths, T) if lengths is not None else None
        )

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.out_norm(x)


class ConvTransformerEncoder(nn.Module):
    """
    Input:  (T, N, D)
    Output: (T, N, D)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        conv_kernel_size: int = 5,
        conv_layers: int = 2,
        use_sinusoidal_pos_emb: bool = True,
        max_len: int = 200000,
        norm_first: bool = True,
    ) -> None:
        super().__init__()

        assert conv_kernel_size % 2 == 1

        conv_blocks: list[nn.Module] = []
        for _ in range(conv_layers):
            conv_blocks.extend(
                [
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=conv_kernel_size,
                        stride=1,
                        padding=conv_kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.conv = nn.Sequential(*conv_blocks)

        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_sinusoidal_pos_emb=use_sinusoidal_pos_emb,
            max_len=max_len,
            norm_first=norm_first,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError(f"Expected inputs of shape (T, N, D), got {tuple(inputs.shape)}")

        x = inputs.permute(1, 2, 0)   # (N, D, T)
        x = self.conv(x)
        x = x.permute(2, 0, 1)        # (T, N, D)

        x = x + inputs
        x = self.out_norm(x)

        return self.transformer(x, lengths=lengths)

class ConvFrontend1D(nn.Module):
    """
    Input:  (T, N, D_in)
    Output: (T_out, N, D_out)
    """

    def __init__(
        self,
        in_features: int,
        channels: Sequence[int] = (64, 128, 256),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        strides: Sequence[int] = (2, 2, 1),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(channels) == len(kernel_sizes) == len(strides)

        layers: list[nn.Module] = []
        c_in = in_features
        for c_out, k, s in zip(channels, kernel_sizes, strides):
            assert k % 2 == 1, "Use odd kernel sizes for same-style padding"
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=k,
                        stride=s,
                        padding=k // 2,
                    ),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            c_in = c_out

        self.network = nn.Sequential(*layers)
        self.out_features = c_in
        self.strides = tuple(strides)
        self.kernel_sizes = tuple(kernel_sizes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError(f"Expected inputs of shape (T, N, D), got {tuple(inputs.shape)}")

        # (T, N, D) -> (N, D, T)
        x = inputs.permute(1, 2, 0)
        x = self.network(x)
        # (N, D_out, T_out) -> (T_out, N, D_out)
        return x.permute(2, 0, 1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Exact Conv1d length formula applied layer by layer:
        L_out = floor((L_in + 2p - (k - 1) - 1) / s + 1)
        with dilation = 1.
        """
        lengths = input_lengths.clone()
        for k, s in zip(self.kernel_sizes, self.strides):
            p = k // 2
            lengths = torch.div(lengths + 2 * p - (k - 1) - 1, s, rounding_mode="floor") + 1
        return lengths

class CNNTransformerEncoder(nn.Module):
    """
    Input:  (T, N, D_in)
    Output: (T_out, N, d_model)
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 256,
        cnn_channels: Sequence[int] = (64, 128, 256),
        cnn_kernel_sizes: Sequence[int] = (7, 5, 3),
        cnn_strides: Sequence[int] = (2, 2, 1),
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_sinusoidal_pos_emb: bool = True,
        max_len: int = 200000,
        norm_first: bool = True,
    ) -> None:
        super().__init__()

        self.frontend = ConvFrontend1D(
            in_features=in_features,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            dropout=dropout,
        )

        self.proj = nn.Linear(self.frontend.out_features, d_model)

        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_sinusoidal_pos_emb=use_sinusoidal_pos_emb,
            max_len=max_len,
            norm_first=norm_first,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_lengths: bool = False,
    ):
        x = self.frontend(inputs)   # (T_out, N, C_out)
        x = self.proj(x)            # (T_out, N, d_model)

        out_lengths = None
        if lengths is not None:
            out_lengths = self.frontend.output_lengths(lengths)

        x = self.transformer(x, lengths=out_lengths)

        if return_lengths:
            return x, out_lengths
        return x

class CNNEncoder(nn.Module):
    """
    Pure 1D CNN sequence encoder.

    Input:  (T, N, D_in)
    Output: (T_out, N, D_out)
    """

    def __init__(
        self,
        in_features: int,
        channels: Sequence[int] = (64, 128, 256),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        strides: Sequence[int] = (2, 2, 1),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(channels) == len(kernel_sizes) == len(strides)

        layers: list[nn.Module] = []
        c_in = in_features
        for c_out, k, s in zip(channels, kernel_sizes, strides):
            assert k % 2 == 1, "Use odd kernel sizes for same-style padding"
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=k,
                        stride=s,
                        padding=k // 2,
                    ),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            c_in = c_out

        self.network = nn.Sequential(*layers)
        self.out_features = c_in
        self.kernel_sizes = tuple(kernel_sizes)
        self.strides = tuple(strides)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_lengths: bool = False,
    ):
        if inputs.dim() != 3:
            raise ValueError(f"Expected inputs of shape (T, N, D), got {tuple(inputs.shape)}")

        # (T, N, D) -> (N, D, T)
        x = inputs.permute(1, 2, 0)
        x = self.network(x)
        # (N, D_out, T_out) -> (T_out, N, D_out)
        x = x.permute(2, 0, 1)

        out_lengths = None
        if lengths is not None:
            out_lengths = self.output_lengths(lengths)

        if return_lengths:
            return x, out_lengths
        return x

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Exact Conv1d length formula applied layer by layer:
        L_out = floor((L_in + 2p - (k - 1) - 1) / s + 1)
        with dilation = 1.
        """
        lengths = input_lengths.clone()
        for k, s in zip(self.kernel_sizes, self.strides):
            p = k // 2
            lengths = torch.div(
                lengths + 2 * p - (k - 1) - 1,
                s,
                rounding_mode="floor",
            ) + 1
        return lengths

class FeedForwardModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """
    Input:  (T, N, D)
    Output: (T, N, D)
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=1,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, D)
        x = self.layer_norm(x)
        x = x.permute(1, 2, 0)  # (N, D, T)

        x = self.pointwise_conv1(x)  # (N, 2D, T)
        x = nn.functional.glu(x, dim=1)  # (N, D, T)

        x = self.depthwise_conv(x)  # (N, D, T)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (N, D, T)
        x = self.dropout(x)

        x = x.permute(2, 0, 1)  # (T, N, D)
        return x


class ConformerBlock(nn.Module):
    """
    Macaron-style conformer block:
      x = x + 0.5 * FFN(x)
      x = x + MHSA(x)
      x = x + Conv(x)
      x = x + 0.5 * FFN(x)
      x = LayerNorm(x)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ffn1 = FeedForwardModule(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False,
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = ConformerConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.ffn2 = FeedForwardModule(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)

        attn_in = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.self_attn_dropout(attn_out)

        x = x + self.conv_module(x)

        x = x + 0.5 * self.ffn2(x)

        x = self.final_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """
    CNN frontend + projection + stacked conformer blocks.

    Input:  (T, N, D_in)
    Output: (T_out, N, d_model)
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 256,
        cnn_channels: Sequence[int] = (64, 128, 256),
        cnn_kernel_sizes: Sequence[int] = (7, 5, 3),
        cnn_strides: Sequence[int] = (2, 2, 1),
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        use_sinusoidal_pos_emb: bool = True,
        max_len: int = 200000,
    ) -> None:
        super().__init__()

        self.frontend = ConvFrontend1D(
            in_features=in_features,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            dropout=dropout,
        )
        self.proj = nn.Linear(self.frontend.out_features, d_model)

        self.use_sinusoidal_pos_emb = use_sinusoidal_pos_emb
        self.max_len = max_len
        self.d_model = d_model

        if use_sinusoidal_pos_emb:
            self.pos_emb = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout,
            )
        else:
            self.pos_emb = None

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def _make_padding_mask(self, lengths: torch.Tensor, T: int) -> torch.Tensor:
        return torch.arange(T, device=lengths.device)[None, :] >= lengths[:, None]

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_lengths: bool = False,
    ):
        x = self.frontend(inputs)   # (T_out, N, C_out)
        x = self.proj(x)            # (T_out, N, d_model)

        out_lengths = None
        if lengths is not None:
            out_lengths = self.frontend.output_lengths(lengths)

        T = x.size(0)
        if T > self.max_len:
            raise ValueError(
                f"T={T} exceeds max_len={self.max_len}. Increase max_len in ConformerEncoder."
            )

        if self.pos_emb is not None:
            x = self.pos_emb(x)

        key_padding_mask = (
            self._make_padding_mask(out_lengths, T)
            if out_lengths is not None
            else None
        )

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.out_norm(x)

        if return_lengths:
            return x, out_lengths
        return x

class LSTMEncoder(nn.Module):
    """
    Input:  (T, N, num_features)
    Output: (T, N, hidden_size * num_directions) if no projection
            (T, N, output_size) if projection is used
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        output_size: int | None = None,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * self.num_directions

        self.proj = None
        if output_size is not None and output_size != lstm_out_size:
            self.proj = nn.Linear(lstm_out_size, output_size)
            self.norm = nn.LayerNorm(output_size)
            self.output_size = output_size
        else:
            self.norm = nn.LayerNorm(lstm_out_size)
            self.output_size = lstm_out_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, C)
        x, _ = self.lstm(inputs)  # (T, N, H * directions)

        if self.proj is not None:
            x = self.proj(x)

        x = self.norm(x)
        return x