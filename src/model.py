"""
DeepSDF model with optional Fourier positional encoding.

Architecture: 8-layer MLP (512 hidden), skip connection at layer 4.
Two variants:
  - Baseline: input = [z; x,y,z]  (latent_dim + 3)
  - With PE:  input = [z; gamma(x,y,z)]  (latent_dim + 6*L)
"""

import math
import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """Fourier positional encoding for 3D coordinates.

    gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p), ...,
                sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]

    Input:  (*, 3)
    Output: (*, 3 * 2L)
    """

    def __init__(self, levels: int = 6):
        super().__init__()
        self.levels = levels
        self.output_dim = 3 * 2 * levels

        # Precompute frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freqs = torch.tensor([2.0 ** k * math.pi for k in range(levels)])
        self.register_buffer("freqs", freqs)  # (L,)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (*, 3) input coordinates
        Returns:
            encoded: (*, 3 * 2L) positional encoding
        """
        # coords: (*, 3) -> (*, 3, 1) * (L,) -> (*, 3, L)
        scaled = coords.unsqueeze(-1) * self.freqs  # (*, 3, L)
        # Interleave sin and cos: (*, 3, 2L)
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        # Flatten last two dims: (*, 6L)
        return encoded.reshape(*coords.shape[:-1], self.output_dim)


class DeepSDF(nn.Module):
    """DeepSDF 8-layer MLP with skip connection at layer 4.

    Architecture follows Park et al. (CVPR 2019):
      - Layers 1-4: Linear + ReLU
      - Skip connection: layer 5 input = layer 4 output + original input
      - Layers 5-8: Linear + ReLU (except last layer: Linear + Tanh)

    Args:
        latent_dim: dimension of per-shape latent code
        hidden_dim: hidden layer width (default 512)
        num_layers: number of MLP layers (default 8)
        skip_layer: layer index for skip connection (default 4, 0-indexed)
        use_pe: whether to use Fourier positional encoding
        pe_levels: number of PE frequency levels (L)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        skip_layer: int = 4,
        use_pe: bool = False,
        pe_levels: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_layer = skip_layer
        self.use_pe = use_pe

        # Positional encoding
        if use_pe:
            self.pe = FourierPositionalEncoding(levels=pe_levels)
            coord_dim = self.pe.output_dim  # 6L
        else:
            self.pe = None
            coord_dim = 3

        input_dim = latent_dim + coord_dim
        self.input_dim = input_dim

        # Build MLP layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_features = input_dim
            elif i == skip_layer:
                in_features = hidden_dim + input_dim  # skip connection
            else:
                in_features = hidden_dim

            if i == num_layers - 1:
                out_features = 1  # SDF output
            else:
                out_features = hidden_dim

            layers.append(nn.Linear(in_features, out_features))

        self.layers = nn.ModuleList(layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Geometric initialization for SDF networks.

        Interior layers use default init. Last layer gets small weights
        and a positive bias to initialize as approximate sphere.
        """
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                # Last layer: small weights, positive bias -> initial SDF ~ sphere
                nn.init.normal_(layer.weight, mean=0.0, std=0.0001)
                nn.init.constant_(layer.bias, 0.1)
            else:
                # Xavier init for interior layers
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, latent_codes: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_codes: (B, latent_dim) per-shape latent vectors
            coords: (B, 3) query point coordinates

        Returns:
            sdf: (B, 1) predicted SDF values
        """
        # Apply positional encoding if enabled
        if self.pe is not None:
            coord_features = self.pe(coords)
        else:
            coord_features = coords

        # Concatenate latent code and coordinate features
        x = torch.cat([latent_codes, coord_features], dim=-1)  # (B, input_dim)
        input_x = x  # save for skip connection

        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                x = torch.cat([x, input_x], dim=-1)  # skip connection

            x = layer(x)

            if i < self.num_layers - 1:
                x = torch.relu(x)

        return x  # (B, 1), raw SDF prediction (no activation on output)


class LatentCodes(nn.Module):
    """Learnable per-shape latent codes for auto-decoder framework.

    Each shape gets a latent vector z ~ N(0, 0.01^2) that is jointly
    optimized with the network weights during training.
    """

    def __init__(self, num_shapes: int, latent_dim: int = 256):
        super().__init__()
        self.num_shapes = num_shapes
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(num_shapes, latent_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B,) shape indices

        Returns:
            codes: (B, latent_dim) latent vectors
        """
        return self.embedding(indices)

    def get_all_codes(self) -> torch.Tensor:
        """Return all latent codes as a tensor. Useful for regularization."""
        return self.embedding.weight
