from abc import ABC, abstractmethod
from typing import Any
import torch


class Preconditioner(ABC):
    def __init__(self, device: torch.device):
        """
        Base class for a preconditioner.

        Args:
            device (torch.device): The device on which computations will be performed.
        """
        self.device = device

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """
        Update the preconditioner based on the provided arguments.
        Subclasses should define the specific arguments required.
        """
        pass

    @abstractmethod
    def inv_lin_op(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse linear operator of the preconditioner to a vector.

        Args:
            v (torch.Tensor): Input vector.

        Returns:
            torch.Tensor: Output vector after applying the preconditioner.
        """
        pass
