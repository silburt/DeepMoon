from torch.nn import (Module, Conv2d, ReLU, PReLU, ELU, Sequential, Dropout2d,
                      ConvTranspose2d)
from torch.nn.modules.batchnorm import _BatchNorm
from torch import (cat, add)
from torch.nn.functional import batch_norm
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import pytorch_lightning as pl

def passthrough(x, **kwargs):
    return x


def Activation(activation, nchan):
    if "elu" == activation:
        return ELU(inplace=True)
    elif "prelu" == activation:
        return PReLU(nchan)
    else:
        return ReLU(inplace=True)


# normalization between sub_volumes.
class ContBatchNorm2d(_BatchNorm):
    def _check_input_dim(self, idata):
        if idata.dim() != 4:
            raise ValueError(f"expected 4d input. I got {idata.dim()}")

    def forward(self, idata):
        self._check_input_dim(idata)
        return batch_norm(idata, self.running_mean, self.running_var,
                          self.weight, self.bias, True, self.momentum,
                          self.eps)


class LUConv(Module):
    def __init__(self, nchan, relu):
        super().__init__()
        self.relu1 = Activation(relu, nchan)
        self.conv1 = Conv2d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(nchan)

    def forward(self, idata):
        idata = self.relu1(self.bn1(self.conv1(idata)))
        return idata


def _make_Conv(nchan, depth, relu):
    layers = list()
    for _ in range(depth):
        layers.append(LUConv(nchan, relu))
    return Sequential(*layers)


class InputTransition(Module):
    def __init__(self, relu):
        super().__init__()

        self.conv1 = Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(16)
        self.relu1 = Activation(relu, 16)

    def forward(self, idata):
        outdata = self.relu1(self.bn1(self.conv1(idata)))

        return outdata


#start downpath
class DownTransition(Module):
    def __init__(self, inchan, nConvs, relu, dropout=0):
        super().__init__()

        outchan = 2 * inchan
        self.down_conv = Conv2d(inchan, outchan, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm2d(outchan)
        self.do1 = passthrough
        self.relu1 = Activation(relu, outchan)
        self.relu2 = Activation(relu, outchan)
        #self.max = MaxPool2d((2,2), stride=(2,2))

        if dropout > 0:
            self.do1 = Dropout2d(dropout)

        self.ops = _make_Conv(outchan, nConvs, relu)

    def forward(self, idata):
        down = self.relu1(self.bn1(self.down_conv(idata)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(add(out, down))
        return out


class UpTransition(Module):
    def __init__(self, inchan, outchan, nConvs, relu, dropout=0):
        super().__init__()

        self.up_conv = ConvTranspose2d(inchan,
                                       outchan // 2,
                                       kernel_size=2,
                                       stride=2)
        self.bn1 = ContBatchNorm2d(outchan // 2)
        self.do1 = passthrough
        self.do2 = Dropout2d(dropout)
        self.relu1 = Activation(relu, outchan // 2)
        self.relu2 = Activation(relu, outchan)

        if dropout > 0:
            self.do1 = Dropout2d(dropout)

        self.ops = _make_Conv(outchan, nConvs, relu)

    def forward(self, idata, skipx):
        up = self.do1(idata)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(up)))
        xcat = cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(add(out, xcat))
        return out


class OutTransition(Module):
    def __init__(self, inchan, relu):
        super().__init__()

        self.conv1 = Conv2d(inchan, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(2)
        self.conv2 = Conv2d(2, 2, kernel_size=1)
        self.relu1 = Activation(relu, 2)
        self.conv3 = Conv2d(2, 1, kernel_size=1)

    def forward(self, idata):
        out = self.relu1(self.bn1(self.conv1(idata)))
        out = self.relu1(self.conv2(out))
        out = self.conv3(out)
        return out


class Crater_VNet(pl.LightningModule):
    def __init__(self, relu="relu", dropout=.15, lr=0.02):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = BCEWithLogitsLoss()
        self.lerning_rate = lr

        self.in_tr = InputTransition(relu)
        self.down_32 = DownTransition(16, 1, relu)
        self.down_64 = DownTransition(32, 1, relu)
        self.down_128 = DownTransition(64, 2, relu, dropout)
        self.down_256 = DownTransition(128, 2, relu, dropout)
        self.up_256 = UpTransition(256, 256, 2, relu, dropout)
        self.up_128 = UpTransition(256, 128, 2, relu, dropout)
        self.up_64 = UpTransition(128, 64, 1, relu)
        self.up_32 = UpTransition(64, 32, 1, relu)
        self.out_tr = OutTransition(32, relu)

    def forward(self, idata):
        out16 = self.in_tr(idata)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        out = self.up_256(out256, out128)
        out = self.up_128(out, out64)
        out = self.up_64(out, out32)
        out = self.up_32(out, out16)
        out = self.out_tr(out)

        return out

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lerning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # data to device
        x, y = train_batch
        
        x_hat = self(x)
        loss = self.criterion(x_hat, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x_hat = self(x)
        loss = self.criterion(x_hat, y)

        self.log('val_loss', loss)
