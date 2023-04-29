""" Dataset module """

import os
import typing as tp
import pandas as pd

import albumentations as albu
import cv2
import numpy as np
import torch
from src.data.preprocessing import (
    augmentation_transforms,
    post_transform,
    pre_transform
)
from torch.utils.data import Dataset
from tqdm.auto import tqdm

DATA_PATH = os.environ["PROJECT_DIR"] + "data/"


class CelebADataset(Dataset):
    def __init__(
        self,
        basic_transforms: albu.BaseCompose,
        data_path: str = DATA_PATH,
        data_type: str = "train",
        sampling_type: str = "augmented",
        augmentation_transforms: tp.Optional[albu.BaseCompose] = None
    ) -> None:
        super().__init__()
        self.data_type = data_type
        self.data_path = data_path
        self.sampling_type = sampling_type

        self.basic_transforms = basic_transforms
        self.augmentation_transforms = augmentation_transforms

        self.label2images_names = dict()
        self.label2count = dict()
        self.label2idx = dict()
        self.idx2label = dict()

        self.image_names = []
        self.labels = []

        self._initialize()
        self.num_of_classes = len(self.label2count.keys())

    def get_batch(
        self,
        n_way: int
    ) -> torch.Tensor:
        image_names = self._get_sample(n_way)

        if self.sampling_type == "same_class":
            images = torch.cat(
                [
                    torch.cat(
                        [
                            self._get_image(anchor)[None, ...],
                            self._get_image(positive)[None, ...],
                            self._get_image(negative)[None, ...],
                        ],
                        dim=0
                    )[None, ...]
                    for anchor, positive, negative in image_names
                ],
                dim=0
            )
        elif self.sampling_type == "augmented":
            images = torch.cat(
                [
                    torch.cat(
                        [
                            self._get_image(anchor)[None, ...],
                            self._get_image(anchor)[None, ...],
                            self._get_image(negative)[None, ...],
                        ],
                        dim=0
                    )[None, ...]
                    for anchor, _, negative in image_names
                ],
                dim=0
            )
        else:
            raise KeyError(
                f"{self.sampling_type} must be in [same_class, augmented]")

        return images

    def _initialize(self) -> None:

        data_table = pd.read_csv(DATA_PATH+self.data_type+".csv")

        for _, line in tqdm(data_table.iterrows(), desc="Reading images", total=len(data_table)):
            img_name = line["image_id"]
            img_label = line["label"]

            if img_label not in self.label2images_names:
                self.label2images_names[img_label] = [img_name]
                self.label2count[img_label] = 1
            else:
                self.label2images_names[img_label] += [img_name]
                self.label2count[img_label] += 1

        for label, values in self.label2images_names.items():
            for img_name in values:
                self.image_names += [img_name]
                self.labels += [label]

        idx = 0
        for label in self.labels:
            if label not in self.label2idx.keys():
                self.label2idx[label] = idx
                self.idx2label[idx] = label
                idx += 1

    def _get_sample(
        self,
        n_anchor_classes: int
    ) -> np.ndarray:
        random_classes = np.random.choice(
            list(self.label2images_names.keys()),
            size=n_anchor_classes * 2,
            replace=False
        )

        images_names = np.array([
            np.random.choice(
                self.label2images_names[class_name],
                size=2
            ).tolist() for class_name in random_classes[:n_anchor_classes]
        ])

        images_names = np.hstack((
            images_names,
            np.array([
                np.random.choice(
                    self.label2images_names[class_name],
                    size=1
                ).tolist() for class_name in random_classes[n_anchor_classes:]
            ])
        ))

        return images_names

    def _get_image(
        self,
        image_name: str
    ) -> torch.Tensor:
        img = cv2.imread(self.data_path + self.data_type + "/" + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augmentation_transforms is not None:
            img = self.augmentation_transforms(image=img)
        else:
            img = self.basic_transforms(image=img)

        return img["image"]

    def __getitem__(self, index: int) -> tp.Tuple:
        img = self._get_image(
            image_name=self.image_names[index]
        )

        return (img, self.label2idx[self.labels[index]])

    def __len__(self):
        return len(self.image_names)


def get_default_dataset():
    basic_transformation = albu.Compose([
        pre_transform(), post_transform()
    ])

    augmentation_transformation = albu.Compose([
        pre_transform(), augmentation_transforms(), post_transform()
    ])

    dataset = CelebADataset(
        basic_transforms=basic_transformation,
        augmentation_transforms=augmentation_transformation
    )

    return dataset
