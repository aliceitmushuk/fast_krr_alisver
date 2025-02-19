import torch


class MinibatchGenerator:
    def __init__(self, n, bg):
        self.n = n
        self.bg = bg
        self.idx = torch.randperm(n)  # Initial shuffle of indices
        self.current_batch = 0
        self.n_batches = (n + bg - 1) // bg

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.n_batches:
            self.idx = torch.randperm(self.n)  # Reshuffle indices
            self.current_batch = 0

        if self.current_batch < self.n_batches:
            start = self.current_batch * self.bg
            end = min((self.current_batch + 1) * self.bg, self.n)
            self.current_batch += 1
            return self.idx[start:end]
        else:
            raise StopIteration
