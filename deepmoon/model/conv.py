from deepmoon.model.activations import Activation
from torch.nn import (Module, Conv2d, Sequential)


class ConvBlock(Module):
    def __init__(self, nchan, relu):
        super().__init__()
        self.conv1 = Conv2d(nchan, nchan, kernel_size=5, padding=2)
        self.activation = Activation(relu, nchan)

    def forward(self, idata):
        idata = self.activation(self.conv1(idata))
        return idata