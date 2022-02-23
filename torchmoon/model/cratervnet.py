from typing import (Any, List)

from torch.nn import (Module, Conv2d, Dropout2d, ConvTranspose2d)
from torch import (cat, add)
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import pytorch_lightning as pl

from torchmetrics import (MaxMetric, StructuralSimilarityIndexMeasure)

from torchmoon.torch.activations import Activation
from torchmoon.torch.util import (passthrough, ContBatchNorm2d)
from torchmoon.torch.conv import make_Conv


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

        if dropout > 0:
            self.do1 = Dropout2d(dropout)

        self.ops = make_Conv(outchan, nConvs, relu)

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

        self.ops = make_Conv(outchan, nConvs, relu)

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

    def __init__(self, activation="relu", dropout=.15, lr=0.02):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.criterion = BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = StructuralSimilarityIndexMeasure()
        self.val_acc = StructuralSimilarityIndexMeasure()
        self.test_acc = StructuralSimilarityIndexMeasure()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.in_tr = InputTransition(self.hparams.activation)
        self.down_32 = DownTransition(16, 1, self.hparams.activation)
        self.down_64 = DownTransition(32, 1, self.hparams.activation)
        self.down_128 = DownTransition(64, 2, self.hparams.activation,
                                       self.hparams.dropout)
        self.down_256 = DownTransition(128, 2, self.hparams.activation,
                                       self.hparams.dropout)
        self.up_256 = UpTransition(256, 256, 2, self.hparams.activation,
                                   self.hparams.dropout)
        self.up_128 = UpTransition(256, 128, 2, self.hparams.activation,
                                   self.hparams.dropout)
        self.up_64 = UpTransition(128, 64, 1, self.hparams.activation)
        self.up_32 = UpTransition(64, 32, 1, self.hparams.activation)
        self.out_tr = OutTransition(32, self.hparams.activation)

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
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def step(self, batch: Any):
        x, y, _ = batch

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        return loss, y_hat, y

    def training_step(self, train_batch: Any, batch_idx: int):
        # data to device
        loss, preds, targets = self.step(train_batch)

        acc = self.train_acc(preds, targets)
        self.log("train/loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, val_batch: Any, batch_idx: int):
        loss, preds, targets = self.step(val_batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best",
                 self.val_acc_best.compute(),
                 on_epoch=True,
                 prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()
