from torch import cat
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.functional import batch_norm

class ContBatchNorm2d(_BatchNorm):
    def _check_input_dim(self, idata):
        if idata.dim() != 4:
            raise ValueError(f"expected 4d input. I got {idata.dim()}")

    def forward(self, idata):
        self._check_input_dim(idata)
        return batch_norm(idata, self.running_mean, self.running_var,
                          self.weight, self.bias, True, self.momentum,
                          self.eps)

def merge(layers, cat_axis=None):
    return cat(layers, dim=cat_axis)

def passthrough(x, **kwargs):
    return x
