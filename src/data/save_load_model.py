from torch import nn, save, load
import datetime
import os

MODELS_PATH = os.environ["PROJECT_DIR"] + "/models/"


def save_model(
    model: nn.Module,
    model_name: str
) -> None:
    time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")

    if not os.path.exists(MODELS_PATH + model_name):
        os.makedirs(MODELS_PATH + model_name)

    save(
        model,
        MODELS_PATH + model_name + "/" + time
    )


def load_model(
    model_name: str,
    load_latest: bool = True
) -> nn.Module:

    if not load_latest:
        return load(MODELS_PATH + model_name)

    return load(
        MODELS_PATH + model_name + "/" +
        os.listdir(MODELS_PATH + model_name + "/")[-1]
    )
