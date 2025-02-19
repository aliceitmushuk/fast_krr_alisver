from abc import ABC, abstractmethod
from typing import Any
import torch


class Preconditioner(ABC):
    """
    Abstract base class for preconditioners used in optimization algorithms.

    A preconditioner is a linear operator applied to modify the gradient or
    parameter update rule, often to accelerate convergence. This class provides
    an interface for implementing custom preconditioners.

    Attributes:
        device (torch.device): The device (CPU or GPU) where computations
          will be performed.
    """

    def __init__(self, device: torch.device):
        """
        Initializes the preconditioner.

        Args:
            device (torch.device): The device on which computations will be performed
                (e.g., torch.device('cpu') or torch.device('cuda')).
        """
        self.device = device

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """
        Updates the preconditioner based on the provided arguments.

        This method is intended to adjust the preconditioner's internal state,
        often based on the current iteration of the optimization process or other
        relevant information.

        Subclasses should specify the required arguments and define the update logic.
        """
        pass

    @abstractmethod
    def inv_lin_op(self, v: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse linear operator of the preconditioner to an input vector.

        This operation modifies the input vector to precondition it, often used
        in solving systems or adjusting gradients.

        Args:
            v (torch.Tensor): Input vector to which the inverse linear operator
              is applied.

        Returns:
            torch.Tensor: The preconditioned output vector.
        """
        pass
