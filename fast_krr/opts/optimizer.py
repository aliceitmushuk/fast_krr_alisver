from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract base class for optimizers. Defines the structure for optimizer classes
    that operate on a model and optionally use preconditioning parameters.

    Attributes:
        model: The model to be optimized. This could be any object that represents
            a machine learning model with parameters to optimize.
        precond_params (dict): A dictionary of parameters for preconditioning.
            These parameters can be used to modify the optimization behavior, such as
            scaling gradients or adapting step sizes.

    Methods:
        step(): Abstract method to perform a single optimization step. This method
            must be implemented by subclasses to define the specific optimization logic.
    """

    def __init__(self, model, precond_params: dict):
        """
        Initializes the optimizer with a model and optional preconditioning parameters.

        Args:
            model: The model to be optimized. Typically an object with parameters
                accessible via an attribute like `parameters()` or similar.
            precond_params (dict): A dictionary of preconditioning parameters. These
                can include options like learning rate schedules or scaling factors.
        """
        self.model = model
        self.precond_params = precond_params

    @abstractmethod
    def step(self):
        """
        Performs a single optimization step.

        This method should be implemented by subclasses to update the model parameters
        based on gradients and any additional logic defined by the optimizer.
        """
        pass
