import matplotlib.colors as mcolors
import numpy as np


# Custom compressed normalization (log scale with a shift)
class CompressedLogNorm(mcolors.Normalize):
    def __init__(self, vmin, vmax, compress_factor=0.4):
        super().__init__(vmin=vmin, vmax=vmax, clip=False)
        self.compress_factor = compress_factor

    def __call__(self, value):
        log_value = np.log10(value)
        log_min = np.log10(self.vmin)
        log_max = np.log10(self.vmax)
        proportion = (log_value - log_min) / (log_max - log_min)
        proportion_compressed = (
            1 - self.compress_factor
        ) * proportion + self.compress_factor
        return proportion_compressed
