from torch.nn import (Conv2d, Sequential, Dropout2d, Upsample)
from torch.nn.init import xavier_uniform
from torch.nn.modules.activation import Sigmoid
from torch import (cat, reshape)
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.pooling import MaxPool2d
from torch.optim import Adam
import pytorch_lightning as pl

from deepmoon.model.activations import Activation

class DeepMoon(pl.LightningModule):
    def __init__(self, number_of_filter, filter_length, lmbda, activation="relu", dim = 256, dropout=.15, lr=0.02):
        super().__init__()
        
        self.save_hyperparameters()

        self.criterion = BCEWithLogitsLoss()
        self.lerning_rate = lr
        self.lmbda = lmbda
        self.dropout = dropout
        self.dim = dim

        self.down_0 = Sequential(
            Conv2d( in_channels=1, # gray image
                    out_channels=number_of_filter,
                    kernel_size=(filter_length,filter_length),
                    padding=1),
            Activation(activation, number_of_filter),
            Conv2d( in_channels=number_of_filter, 
                    out_channels=number_of_filter,
                    kernel_size=(filter_length,filter_length),
                    padding=1),
            Activation(activation, number_of_filter),
            MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )

        number_of_filter_2 = number_of_filter * 2
        self.down_1 = Sequential(
            Conv2d( in_channels=number_of_filter, 
                    out_channels=number_of_filter_2,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter),
            Conv2d( in_channels=number_of_filter_2, 
                    out_channels=number_of_filter_2,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter_2),
            MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )

        number_of_filter_4 = number_of_filter * 4
        self.down_2 = Sequential(
            Conv2d( in_channels=number_of_filter_2, 
                    out_channels=number_of_filter_4,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter),
            Conv2d( in_channels=number_of_filter_4, 
                    out_channels=number_of_filter_4,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter_4),
        )

        self.down_3 = Sequential(
            Conv2d( in_channels=number_of_filter_4, 
                    out_channels=number_of_filter_4,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter),
            Conv2d( in_channels=number_of_filter_4, 
                    out_channels=number_of_filter_4,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter_4)
        )

        self.down_0.apply(self.init_weights)
        self.down_1.apply(self.init_weights)
        self.down_2.apply(self.init_weights)
        self.down_3.apply(self.init_weights)

        self.up_1 = Sequential(
            Conv2d( in_channels=number_of_filter_4, 
                    out_channels=number_of_filter_2,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter),
            Conv2d( in_channels=number_of_filter_2, 
                    out_channels=number_of_filter_2,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter_2)
        )

        self.up_2 = Sequential(
            Conv2d( in_channels=number_of_filter_2, 
                    out_channels=number_of_filter,
                    kernel_size=filter_length,
                    padding=1 ),
            Activation(activation, number_of_filter),
            Conv2d( in_channels=number_of_filter, 
                    out_channels=number_of_filter,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter)
        )


        self.up_3 = Sequential(
            Conv2d( in_channels=number_of_filter_2, 
                    out_channels=number_of_filter,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter),
            Conv2d( in_channels=number_of_filter, 
                    out_channels=number_of_filter,
                    kernel_size=filter_length,
                    padding=1),
            Activation(activation, number_of_filter)
        )

        self.up_1.apply(self.init_weights)
        self.up_2.apply(self.init_weights)
        self.up_3.apply(self.init_weights)

        self.out_conv = Conv2d(in_channels=number_of_filter, out_channels=1, kernel_size=2)

        self.out_conv.apply(self.init_weights)

    def forward(self, idata):
        d0 = self.down_0(idata)
        max_1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))(d0)
        d1 = self.down_1(max_1)
        max_2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))(d1)
        d2 = self.down_2(max_2)
        max_3 = MaxPool2d(kernel_size=(2,2), stride=(2,2))(d2)

        u = self.down_3(max_3)
        u = Upsample(size=d2.shape[-2:])(u)
        u = merge(layers=(d2,u), cat_axis=1)
        u = self.dropout_reg(u)

        u = self.up_1(u)
        u = Upsample(size=d1.shape[-2:])(u)
        u = merge(layers=(d1,u), cat_axis=1)
        u = self.dropout_reg(u)

        u = self.up_2(u)
        u = Upsample(size=d0.shape[-2:])(u)
        u = merge(layers=(d0, u), cat_axis=1)
        u = self.dropout_reg(u)

        u = self.up_3(u)

        u = self.out_conv(u)
        u = Sigmoid()(u)

        return reshape(input=u, shape=(self.dim, self.dim))

    def dropout_reg(self, u):
        return Dropout2d(p=self.dropout, inplace=True)(u) if self.dropout is not None and self.dropout > 0 else u

    def init_weights(self, m):
        if isinstance(m, Conv2d):
            xavier_uniform(m.weight)
            #m.bias.data.fill_(self.lmbda)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lerning_rate, weight_decay=self.lmbda)
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