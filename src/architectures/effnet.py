""" Efficient module """

import typing as tp

from torch import nn
from torchvision.models import efficientnet_v2_m


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class EfficientNetModelB3(nn.Module):
    def __init__(
        self,
        embedding_size: int,
    ) -> None:
        super().__init__()
        self.model = efficientnet_v2_m(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Identity()
        self.embedder = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.SiLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.SiLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=embedding_size,
                      out_features=embedding_size),
            nn.SiLU(),
            nn.Linear(in_features=embedding_size, out_features=30),
        )
        self.embedder.apply(init_weights)
        self.classifier.apply(init_weights)

    def set_reauires_grad(self) -> None:
        for _, param in self.model.named_parameters():
            param.requires_grad = True

        for _, param in self.embedder.named_parameters():
            param.requires_grad = True

        for _, param in self.classifier.named_parameters():
            param.requires_grad = True

    def set_model_head_requires_grad(self) -> None:
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        for _, param in self.embedder.named_parameters():
            param.requires_grad = True

        for _, param in self.classifier.named_parameters():
            param.requires_grad = True

    def set_model_to_finetune(self) -> None:
        self.model.eval()
        self.embedder.train()
        self.classifier.train()

    def get_trainable_params(self) -> tp.List:
        trainable_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                trainable_params += [param]

        for param in self.embedder.parameters():
            if param.requires_grad:
                trainable_params += [param]

        for param in self.classifier.parameters():
            if param.requires_grad:
                trainable_params += [param]

        return trainable_params

    def forward(self, img) -> tp.Dict:
        model_out = self.model(img)
        embedding = self.embedder(model_out)
        logits = self.classifier(embedding)

        return {
            "embedding": embedding,
            "logits": logits
        }
