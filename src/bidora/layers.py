"""Custom BiDoRA layers with magnitude-direction decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BiDoRALinear(nn.Module):
    """
    BiDoRA Linear Layer with magnitude-direction decomposition.
    
    Implements weight decomposition: W' = m * (W0 + BA) / ||W0 + BA||
    where:
        - m: magnitude component (optimized at upper level)
        - W0 + BA: direction component (optimized at lower level)
        - B, A: low-rank matrices
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Base weight (frozen during training)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Low-rank matrices (direction component)
        self.lora_A = nn.Parameter(torch.empty((rank, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, rank)))

        # Magnitude component (optimized at upper level)
        self.magnitude = nn.Parameter(torch.ones(out_features))

        # Dropout
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Scaling factor
        self.scaling = lora_alpha / rank if rank > 0 else 1.0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters following Kaiming initialization."""
        # Base weight: Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # LoRA matrices: Following LoRA paper
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Magnitude: Initialize to column-wise norm of base weight
        with torch.no_grad():
            self.magnitude.data = self.weight.norm(p=2, dim=0, keepdim=False)

    def get_direction_weights(self) -> Tensor:
        """
        Compute direction component: (W0 + BA) / ||W0 + BA||
        
        Returns:
            Direction weights (normalized)
        """
        # Low-rank update
        delta_w = self.lora_B @ self.lora_A * self.scaling

        # Combined weight
        w_combined = self.weight + delta_w

        # Normalize column-wise
        w_norm = w_combined.norm(p=2, dim=0, keepdim=True).clamp(min=1e-8)
        direction = w_combined / w_norm

        return direction

    def get_bidora_weights(self) -> Tensor:
        """
        Compute full BiDoRA weights: m * direction
        
        Returns:
            BiDoRA weights with magnitude scaling
        """
        direction = self.get_direction_weights()
        # Apply magnitude scaling
        weights = direction * self.magnitude.unsqueeze(0)
        return weights

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using BiDoRA weights.
        
        Args:
            x: Input tensor [batch_size, in_features]
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        x = self.dropout(x)

        # Get BiDoRA weights
        weights = self.get_bidora_weights()

        # Linear transformation
        output = F.linear(x, weights, self.bias)

        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"lora_alpha={self.lora_alpha}, "
            f"bias={self.bias is not None}"
        )


def replace_linear_with_bidora(
    module: nn.Module,
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> None:
    """
    Replace Linear layers with BiDoRALinear in-place.

    Args:
        module: Root module to process
        rank: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout rate
        target_modules: List of module names to replace (e.g., ['q_proj', 'v_proj'])
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    for name, child in module.named_children():
        # Check if this is a Linear layer (nn.Linear or quantized Linear4bit/Linear8bitLt)
        is_linear = isinstance(child, nn.Linear) or (
            hasattr(child, "in_features")
            and hasattr(child, "out_features")
            and type(child).__name__ in ("Linear4bit", "Linear8bitLt")
        )

        # Check if this is a target module
        if is_linear and any(target in name for target in target_modules):
            # Create BiDoRA layer
            bidora_layer = BiDoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=child.bias is not None if hasattr(child, "bias") else False,
            )

            # Match dtype of original layer
            target_dtype = child.weight.dtype
            bidora_layer = bidora_layer.to(dtype=target_dtype)

            # Copy base weights
            with torch.no_grad():
                bidora_layer.weight.copy_(child.weight)
                if hasattr(child, "bias") and child.bias is not None and bidora_layer.bias is not None:
                    bidora_layer.bias.copy_(child.bias)

            # Replace module
            setattr(module, name, bidora_layer)

        else:
            # Recursively process children
            replace_linear_with_bidora(
                child, rank, lora_alpha, lora_dropout, target_modules
            )
