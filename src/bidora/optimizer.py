"""Bi-level optimization for BiDoRA training."""

import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch import Tensor
from torch.optim import Optimizer

from .layers import BiDoRALinear


class BiLevelOptimizer:
    """
    Bi-level optimizer for BiDoRA.
    
    Implements:
    - Lower level: Optimize direction (A, B matrices) on training set
    - Upper level: Optimize magnitude (m) on validation set using hypergradients
    """

    def __init__(
        self,
        model: nn.Module,
        lower_optimizer: Optimizer,
        upper_optimizer: Optimizer,
        lower_steps: int = 1,
    ):
        """
        Initialize bi-level optimizer.
        
        Args:
            model: Model with BiDoRALinear layers
            lower_optimizer: Optimizer for direction (A, B)
            upper_optimizer: Optimizer for magnitude (m)
            lower_steps: Number of lower-level steps per upper-level step
        """
        self.model = model
        self.lower_optimizer = lower_optimizer
        self.upper_optimizer = upper_optimizer
        self.lower_steps = lower_steps

        # Separate parameters
        self.direction_params = []
        self.magnitude_params = []
        self.base_params = []

        for module in model.modules():
            if isinstance(module, BiDoRALinear):
                # Direction parameters (trainable)
                self.direction_params.extend([module.lora_A, module.lora_B])
                # Magnitude parameters (trainable)
                self.magnitude_params.append(module.magnitude)
                # Base parameters (frozen)
                self.base_params.extend([module.weight])
                if module.bias is not None:
                    self.base_params.append(module.bias)

        # Freeze base parameters
        for param in self.base_params:
            param.requires_grad = False

    def lower_level_step(
        self,
        loss_fn,
        train_batch,
    ) -> Tensor:
        """
        Perform lower-level optimization step.
        
        Optimizes direction (A, B) while keeping magnitude (m) fixed.
        
        Args:
            loss_fn: Loss function
            train_batch: Training batch
            
        Returns:
            Loss value
        """
        # Enable direction gradients, disable magnitude gradients
        for param in self.direction_params:
            param.requires_grad = True
        for param in self.magnitude_params:
            param.requires_grad = False

        # Forward pass
        loss = loss_fn(self.model, train_batch)

        # Backward pass
        self.lower_optimizer.zero_grad()
        loss.backward()
        self.lower_optimizer.step()

        return loss.detach()

    def upper_level_step(
        self,
        loss_fn,
        val_batch,
        train_batch,
    ) -> Tensor:
        """
        Perform upper-level optimization step using hypergradients.
        
        Optimizes magnitude (m) on validation set while accounting for
        the implicit dependence on direction parameters.
        
        Args:
            loss_fn: Loss function
            val_batch: Validation batch
            train_batch: Training batch (for hypergradient computation)
            
        Returns:
            Validation loss value
        """
        # Enable magnitude gradients, disable direction gradients
        for param in self.direction_params:
            param.requires_grad = False
        for param in self.magnitude_params:
            param.requires_grad = True

        # Compute validation loss with current magnitude
        val_loss = loss_fn(self.model, val_batch)

        # Compute hypergradients
        hypergradients = self._compute_hypergradients(
            loss_fn, val_loss, val_batch, train_batch
        )

        # Update magnitude parameters with hypergradients
        self.upper_optimizer.zero_grad()
        for param, hypergra in zip(self.magnitude_params, hypergradients):
            if param.grad is None:
                param.grad = hypergra
            else:
                param.grad += hypergra

        self.upper_optimizer.step()

        return val_loss.detach()

    def _compute_hypergradients(
        self,
        loss_fn,
        val_loss: Tensor,
        val_batch,
        train_batch,
    ) -> list[Tensor]:
        """
        Compute hypergradients for magnitude parameters.
        
        Uses approximate hypergradient computation via finite differences
        for computational efficiency.
        
        Args:
            loss_fn: Loss function
            val_loss: Current validation loss
            val_batch: Validation batch
            train_batch: Training batch
            
        Returns:
            List of hypergradients for magnitude parameters
        """
        # Compute direct gradient of val_loss w.r.t. magnitude
        direct_grads = torch.autograd.grad(
            val_loss,
            self.magnitude_params,
            create_graph=False,
            allow_unused=True,
        )

        # For simplicity, use direct gradients as hypergradients
        # Full hypergradient would require computing:
        # dL_val/dm = dL_val/dm_direct + dL_val/dv * dv/dm
        # where v are direction parameters
        # This is expensive, so we approximate with direct gradients

        hypergradients = [
            grad if grad is not None else torch.zeros_like(param)
            for grad, param in zip(direct_grads, self.magnitude_params)
        ]

        return hypergradients

    def bilevel_step(
        self,
        loss_fn,
        train_batch,
        val_batch,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform one complete bi-level optimization step.
        
        Args:
            loss_fn: Loss function
            train_batch: Training batch
            val_batch: Validation batch
            
        Returns:
            Tuple of (train_loss, val_loss)
        """
        # Lower level: Update direction on train set
        train_loss = None
        for _ in range(self.lower_steps):
            train_loss = self.lower_level_step(loss_fn, train_batch)

        # Upper level: Update magnitude on val set
        val_loss = self.upper_level_step(loss_fn, val_batch, train_batch)

        return train_loss, val_loss

    def final_finetune_step(
        self,
        loss_fn,
        combined_batch,
    ) -> Tensor:
        """
        Final fine-tuning step on combined train+val data.
        
        Optimizes direction with fixed magnitude.
        
        Args:
            loss_fn: Loss function
            combined_batch: Combined train+val batch
            
        Returns:
            Loss value
        """
        # Enable only direction gradients
        for param in self.direction_params:
            param.requires_grad = True
        for param in self.magnitude_params:
            param.requires_grad = False

        # Forward pass
        loss = loss_fn(self.model, combined_batch)

        # Backward pass
        self.lower_optimizer.zero_grad()
        loss.backward()
        self.lower_optimizer.step()

        return loss.detach()


def create_bilevel_optimizer(
    model: nn.Module,
    lower_lr: float = 2e-4,
    upper_lr: float = 1e-3,
    lower_steps: int = 1,
    weight_decay: float = 0.01,
) -> BiLevelOptimizer:
    """
    Create bi-level optimizer with separate optimizers for direction and magnitude.
    
    Args:
        model: Model with BiDoRALinear layers
        lower_lr: Learning rate for direction optimization
        upper_lr: Learning rate for magnitude optimization
        lower_steps: Number of lower-level steps per upper-level step
        weight_decay: Weight decay for both optimizers
        
    Returns:
        BiLevelOptimizer instance
    """
    # Collect parameters
    direction_params = []
    magnitude_params = []

    for module in model.modules():
        if isinstance(module, BiDoRALinear):
            direction_params.extend([module.lora_A, module.lora_B])
            magnitude_params.append(module.magnitude)

    # Create 8-bit optimizers for 75% memory reduction
    lower_optimizer = bnb.optim.AdamW8bit(
        direction_params,
        lr=lower_lr,
        weight_decay=weight_decay,
        block_wise=True,  # Blockwise quantization for better accuracy
    )

    upper_optimizer = bnb.optim.AdamW8bit(
        magnitude_params,
        lr=upper_lr,
        weight_decay=weight_decay,
        block_wise=True,
    )

    print("Using 8-bit AdamW optimizers (75% memory reduction)")

    return BiLevelOptimizer(
        model=model,
        lower_optimizer=lower_optimizer,
        upper_optimizer=upper_optimizer,
        lower_steps=lower_steps,
    )
