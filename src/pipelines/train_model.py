import os
import typing as tp

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.data.dataset import CelebADataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_ANCHOR_CLASSES = int(os.environ["N_ANCHOR_CLASSES"])


MARGIN = float(os.environ["MARGIN"])
SESSION_SIZE = int(os.environ["SESSION_SIZE"])


IMG_H = int(os.environ["IMG_H"])
IMG_W = int(os.environ["IMG_W"])

MODEL_NAME = os.environ["MODEL_NAME"]

writer = tb.writer.SummaryWriter()


def train_triplet_epoch(
    model: nn.Module,
    train_dataset: Dataset,
    loss_func: nn.TripletMarginLoss,
    opt: torch.optim.Optimizer,
    epoch_numer: int = 0,
    tensorboard: bool = False,
    gradient_clip: tp.Optional[int | None] = None
) -> tp.Dict:
    history = {
        "loss": [],
        "accuracy": []
    }

    for step in tqdm(range(SESSION_SIZE,)):
        model.zero_grad()

        data = train_dataset.get_batch(
            N_ANCHOR_CLASSES).view(-1, 3, IMG_W, IMG_H).to(DEVICE)

        embeddings = model(data)["embedding"].view(
            N_ANCHOR_CLASSES, 3, -1)

        loss = loss_func(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2]
        )

        dists_neg = torch.norm(embeddings[:, 0] - embeddings[:, 2], dim=1)
        dists_pos = torch.norm(embeddings[:, 0] - embeddings[:, 1], dim=1)
        accuracy = torch.mean(
            dists_neg > dists_pos + loss_func.margin,
            dtype=float
        )

        history["loss"] += [loss.item()]
        history["accuracy"] += [accuracy.item()]

        if tensorboard:
            writer.add_scalar(
                f"TRIPLET | Training loss {MODEL_NAME}",
                history["loss"][-1],
                epoch_numer*session_size + step
            )
            writer.add_scalar(
                f"TRIPLET |Training accuracy {MODEL_NAME}",
                history["accuracy"][-1],
                epoch_numer*session_size + step
            )
        loss.backward()

        if gradient_clip is not None:
            clip_grad_norm_(model.parameters(), gradient_clip)

        opt.step()

    return history


def train_classification_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    scheduler,
    loss_func: nn.CrossEntropyLoss,
    opt: torch.optim.Optimizer,
    epoch_numer: int = 0,
    tensorboard: bool = False,
    gradient_clip: tp.Optional[int | None] = None,
    iter_to_stop: int = None
) -> tp.Dict:
    history = {
        "loss": [],
        "accuracy": []
    }

    step = 0
    train_dataloader_len = len(train_dataloader)
    for images, labels in tqdm(train_dataloader):
        model.zero_grad()

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)["logits"]
        loss = loss_func(logits, labels)

        accuracy = torch.mean(
            torch.argmax(logits, dim=1) == labels,
            dtype=float
        )

        history["loss"] += [loss.item()]
        history["accuracy"] += [accuracy.item()]

        if tensorboard:
            writer.add_scalar(
                f"Cls | Training loss {MODEL_NAME}",
                history["loss"][-1],
                epoch_numer*train_dataloader_len + step
            )
            writer.add_scalar(
                f"Cls | Training accuracy {MODEL_NAME}",
                history["accuracy"][-1],
                epoch_numer*train_dataloader_len + step
            )
        loss.backward()

        if gradient_clip is not None:
            clip_grad_norm_(model.parameters(), gradient_clip)

        opt.step()
        scheduler.step()
        step += 1

        if iter_to_stop is not None and iter_to_stop <= step:
            break

    return history


def train_memory_classification_epoch(
    model: nn.Module,
    train_dataset: CelebADataset,
    loss_func: nn.TripletMarginLoss,
    opt: torch.optim.Optimizer,
    session_size: int,
    epoch_numer: int = 0,
    tensorboard: bool = False,
    gradient_clip: tp.Optional[int | None] = None
) -> tp.Dict:
    history = {
        "loss": [],
        "accuracy": []
    }

    for step in tqdm(range(session_size)):
        model.zero_grad()

        data = train_dataset.get_batch(
            N_ANCHOR_CLASSES).view(-1, 3, IMG_W, IMG_H).to(DEVICE)

        embeddings = model(data)["embedding"].view(
            N_ANCHOR_CLASSES, 3, -1)

        loss = loss_func(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2]
        )

        dists_neg = torch.norm(embeddings[:, 0] - embeddings[:, 2], dim=1)
        dists_pos = torch.norm(embeddings[:, 0] - embeddings[:, 1], dim=1)
        accuracy = torch.mean(
            dists_neg > dists_pos + loss_func.margin,
            dtype=float
        )

        history["loss"] += [loss.item()]
        history["accuracy"] += [accuracy.item()]

        if tensorboard:
            writer.add_scalar(
                f"Training loss {MODEL_NAME}",
                history["loss"][-1],
                epoch_numer*session_size + step
            )
            writer.add_scalar(
                f"Training accuracy {MODEL_NAME}",
                history["accuracy"][-1],
                epoch_numer*session_size + step
            )
        loss.backward()

        if gradient_clip is not None:
            clip_grad_norm_(model.parameters(), gradient_clip)

        opt.step()

    return history


def train(
    model: nn.Module,
    dataset: Dataset,
    lr: float,
    batch_size: int,
    epoch_num: int,
    trainable_params: tp.List,
    tensorboard: bool = False,
    gradient_clip: tp.Optional[int | None] = None,
    training_type: str = "classification",
    iter_to_stop: int = None
) -> tp.Dict:
    history = {
        "loss": [],
        "accuracy": []
    }

    model.to(DEVICE)
    opt = optim.Adam(trainable_params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        opt, step_size=50, gamma=0.5, last_epoch=-1)
    triplet_loss = nn.TripletMarginLoss(margin=MARGIN, p=2.0)
    nlll = nn.CrossEntropyLoss()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    for epoch in range(epoch_num):
        if training_type == "classification":
            epoch_hist = train_classification_epoch(
                model,
                dataloader,
                scheduler,
                nlll,
                opt,
                epoch,
                tensorboard,
                gradient_clip,
                iter_to_stop
            )
        elif training_type == "triplet":
            epoch_hist = train_triplet_epoch(
                model,
                dataset,
                triplet_loss,
                opt,
                epoch,
                tensorboard,
                gradient_clip
            )
        else:
            raise ValueError("Wrong training_type was provided")

        history["loss"].extend(epoch_hist["loss"])
        history["accuracy"].extend(epoch_hist["accuracy"])

    return history
