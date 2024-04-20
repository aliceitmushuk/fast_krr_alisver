from abc import ABC, abstractmethod


class Kernel(ABC):
    def __init__(self, x1_lazy, x2_lazy, kernel_params):
        self.K = self._compute_kernel(x1_lazy, x2_lazy, kernel_params)

    @staticmethod
    @abstractmethod
    def _check_kernel_params(kernel_params):
        """Check the kernel parameters."""
        pass

    @abstractmethod
    def _compute_kernel(self, x1_lazy, x2_lazy, kernel_params):
        """Compute the kernel between x1 and x2 with given parameters."""
        pass

    @abstractmethod
    def get_diag(self):
        """Return the diagonal of the kernel matrix."""
        pass

    @abstractmethod
    def get_trace(self):
        """Return the trace of the kernel matrix."""
        pass

    def __matmul__(self, v):
        """Handle the matrix multiplication operator @ for Kernel instances."""
        return self.K @ v

    def __getattr__(self, name):
        """Handle attribute access for attributes not explicitly defined."""
        if name == "T":
                return self.K.T
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")
