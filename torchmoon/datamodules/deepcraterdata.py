from typing import (Tuple, Optional)

from math import floor

from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from pytorch_lightning import LightningDataModule

from craterdata.mooncraterdataset import MoonCraterDataset


class CraterDataModule(LightningDataModule):

    def __init__(self,
                 data_dir: str = "data/",
                 batch_size: int = 256,
                 num_worker: int = 8,
                 image_size: int = 256,
                 training_test_eval_split: Tuple[float, float,
                                                 float] = (0.25, 0.25, 0.75),
                 download: bool = False,
                 pin_memory: bool = False) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transform = transforms.Compose([
            transforms.Resize([self.hparams.image_size, self.hparams.image_size]),
            transforms.ToTensor(),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.crater_dataset = None

    def prepare_data(self):
        if self.crater_dataset is None:
            self.crater_dataset = MoonCraterDataset(
                root=self.hparams.data_dir,
                transform=self.transform,
                target_transform=self.transform,
                download=self.hparams.download)

    def setup(self, stage: Optional[str] = None):

        train_val_test_split = [
            floor(value * len(self.crater_dataset))
            for value in self.hparams.training_test_eval_split
        ]

        if self.data_train is None and self.data_val is None and self.data_test is None:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.crater_dataset,
                lengths=train_val_test_split,
                generator=Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_worker,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_worker,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_worker,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)
