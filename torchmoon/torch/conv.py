from torch.nn import (Module, Conv2d, Sequential)

from torchmoon.torch.activations import Activation
from torchmoon.torch.util import ContBatchNorm2d

class LUConv(Module):
    def __init__(self, nchan, relu):
        super().__init__()
        self.relu1 = Activation(relu, nchan)
        self.conv1 = Conv2d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(nchan)

    def forward(self, idata):
        idata = self.relu1(self.bn1(self.conv1(idata)))
        return idata


def make_Conv(nchan, depth, relu):
    layers = list()
    for _ in range(depth):
        layers.append(LUConv(nchan, relu))
    return Sequential(*layers)


class ConvBlock(Module):
    def __init__(self, ichan, ochan, relu):
        super().__init__()
        self.conv1 = Conv2d(ichan, ochan, kernel_size=5, padding=2)
        self.activation = Activation(relu, ochan)

    def forward(self, idata):
        idata = self.activation(self.conv1(idata))
        return idata
