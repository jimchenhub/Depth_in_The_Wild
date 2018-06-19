import numpy as np
import torch


class Metrics():
    def __init__(self, output, target):
        self.abs_diff = (output - target).abs()

        self.mse = (torch.pow(abs_diff, 2)).mean()
        self.rmse = math.sqrt(self.mse)
        self.mae = abs_diff.mean()
        self.lg10 = (log10(output) - log10(target)).abs().mean()
        self.absrel = (abs_diff / target).mean()

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = (maxRatio < 1.25).float().mean()
        self.delta2 = (maxRatio < 1.25 ** 2).float().mean()
        self.delta3 = (maxRatio < 1.25 ** 3).float().mean()

    def show_metrics(self):
        print("""
            mse: %f,
            rmse: %f,
            mae: %f,
            mae(log): %f,
            <1.25: %f,
            <1.25^2: %f,
            <1.25^3: %f
            """)