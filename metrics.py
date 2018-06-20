import math
import numpy as np
import torch


class Metrics():
    def single_metrics(self, output, target):
        abs_diff = (output - target).abs()
        mse = (torch.pow(abs_diff, 2)).mean()
        rmse = math.sqrt(mse)
        mae = abs_diff.mean()
        # lg10 = (np.log10(output) - np.log10(target)).abs().mean()
        absrel = (abs_diff / target).mean()
        # percentage
        maxRatio = torch.max(output / target, target / output)
        delta1 = (maxRatio < 1.25).float().mean()
        delta2 = (maxRatio < 1.25 ** 2).float().mean()
        delta3 = (maxRatio < 1.25 ** 3).float().mean()

        return mse, rmse, mae, delta1, delta2, delta3

    def calculate(self, outputs, targets):
        result = np.zeros(6)
        for output, target in zip(outputs, targets):
            mse, rmse, mae, delta1, delta2, delta3 = self.single_metrics(output, target)
            result[0] += mse
            result[1] += rmse
            result[2] += mae 
            result[3] += delta1
            result[4] += delta2
            result[5] += delta3
        result /= len(outputs)
        return result

    def show_metrics(self, outputs, targets):
        result = self.calculate(outputs, targets)        
        print("mse:       %f" % result[0])
        print("rmse:      %f" % result[1])
        print("mae:       %f" % result[2])
        # print("mae(log):  %f" % lg10)
        print("<1.25:     %f" % result[3])
        print("<1.25^2:   %f" % result[4])
        print("<1.25^3:   %f" % result[5])


if __name__ == '__main__':
    output = torch.rand(3, 4)
    target = torch.rand(3, 4)
    me = Metrics()
    me.show_metrics([output], [target])