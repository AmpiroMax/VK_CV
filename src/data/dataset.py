import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import pandas as pd
import albumentations as alb
DATA_PATH = "data/"


class SportDataset(Dataset):
    def __init__(self, type: str, transforms: alb.BasicTransform) -> None:
        super().__init__()

        data = pd.read_csv(DATA_PATH + "train.csv")
