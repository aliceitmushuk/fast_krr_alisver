import matplotlib.colors as mcolors
import numpy as np


class CompressedRootNorm(mcolors.Normalize):
    def __init__(self, vmin, vmax, root=2):
        super().__init__(vmin=vmin, vmax=vmax, clip=False)
        self.root = root

    def __call__(self, value):
        scaled_value = (value - self.vmin) / (self.vmax - self.vmin)  # Scale to [0, 1]
        return np.power(scaled_value, 1 / self.root)
