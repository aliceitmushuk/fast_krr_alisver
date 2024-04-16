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
