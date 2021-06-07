from pathlib import Path
from typing import Optional, Callable
from PIL import Image

import numpy as np
import json
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MoonCrater(Dataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 image_size=256) -> None:
        root = Path(root)
        self.transform = transform

        self.img_size = (image_size, image_size)

        #data_file=root / "data_rec.json"
        #assert data_file.is_file(), f"file: {data_file.is_file()} not found"
        #with open(data_file, "r", encoding="utf8") as jsonfile:
        #        info = list(json.load(jsonfile))
        
        files = set()

        for element in (root / "mask").glob("*.png"):
            index = int(element["name"])
            image = root / f"{index}.png"
            mask = root / "mask" / f"{index}.png"
            if image.is_file() and mask.is_file() :
                files.add(tuple((image, mask)))

        self.files = tuple(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()

        data = self.files[index]

        image = Image.open(data[0])
        mask = Image.open(data[1])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return (image, mask)

class MoonCraterH5(Dataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None) -> None:
        root = Path(root)
        self.transform = transform
        
        self.h5file = h5py.File(root, "r")
        self.images = self.h5file["image"]
        self.masks = self.h5file["mask"]

    def __del__(self):
        self.h5file.close()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()

        image = Image.fromarray(self.images[index,...])
        mask = Image.fromarray(self.masks[index,...])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return (image, mask)

