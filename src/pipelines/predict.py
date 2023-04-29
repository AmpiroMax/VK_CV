import os

import albumentations as albu
import cv2
import pandas as pd
import torch

from src.data.preprocessing import post_transform, pre_transform
from src.data.save_load_model import load_model
from src.data.dataset import get_default_dataset

from tqdm.auto import tqdm

DATA_PATH = os.environ["PROJECT_DIR"] + "data/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model_name: str):
    dataset = get_default_dataset()

    model = load_model(
        model_name
    ).to(DEVICE)
    model.eval()

    basic_transformation = albu.Compose([
        pre_transform(), post_transform()
    ])

    test = pd.read_csv(DATA_PATH + "test.csv")
    test["label"] = dataset.idx2label[0]

    for idx, row in tqdm(test.iterrows(), total=len(test)):
        img = cv2.imread(DATA_PATH + "test/" + row["image_id"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = basic_transformation(image=img)["image"].to(DEVICE)

        with torch.no_grad():
            predicted_idx = torch.argmax(model(img[None, ...])["logits"])

        label = dataset.idx2label[predicted_idx.item()]
        test.loc[idx, "label"] = label

    test.to_csv(DATA_PATH + "submit.csv", index=False)
