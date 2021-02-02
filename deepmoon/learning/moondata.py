from pathlib import Path
from typing import Optional, Callable
from PIL import Image

import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MoonCrater(Dataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 image_size=256) -> None:
        self.root = Path(root)
        self.transform = transform

        self.img_size = (image_size, image_size)

        with open(self.root / "data_rec.json", "r", encoding="utf8") as jsonfile:
            self.info = tuple(json.load(jsonfile))

    def __len__(self) -> int:
        return len(self.info)

    def __getitem__(self, index: int) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()

        image = Image.open(self.root / f"{index}.png")
        mask = Image.open(self.root / "mask" / f"{index}.png")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return (image, mask)

