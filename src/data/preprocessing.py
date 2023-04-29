""" Module of image transformations """

import os

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

IMG_H = int(os.environ["IMG_H"])
IMG_W = int(os.environ["IMG_W"])


def pre_transform() -> albu.BasicTransform:
    """
    Creating preporation transformation before augmentation
    """
    return albu.Resize(IMG_H, IMG_W, always_apply=True)


def augmentation_transforms() -> albu.BaseCompose:
    """
    Creating augmenting transformation
    """
    result = [
        albu.GaussNoise(),
        albu.OneOf(
            [
                albu.MotionBlur(blur_limit=3, p=0.7),
                albu.MedianBlur(blur_limit=3, p=1.0),
                albu.Blur(blur_limit=3, p=0.7),
            ],
            p=0.5
        ),
        albu.OneOf(
            [
                albu.RandomGamma(gamma_limit=(85, 115), p=0.5),
                albu.RandomBrightnessContrast(
                    brightness_limit=0.5,
                    contrast_limit=0.5,
                    p=0.5
                ),
                albu.CLAHE(clip_limit=2),
                # albu.ToGray(p=0.5)
            ],
            p=0.5
        ),
        albu.Rotate(
            limit=5,
            crop_border=True
        ),
        albu.HorizontalFlip(),
        albu.RandomResizedCrop(
            height=IMG_H,
            width=IMG_W,
            scale=(0.75, 0.75),
            always_apply=True
        ),
        # albu.augmentations.dropout.CoarseDropout(
        #     max_holes=10,
        #     max_height=IMG_H//10,
        #     max_width=IMG_W//10,
        #     p=0.5
        # )
    ]
    return albu.Compose(result)


def post_transform() -> albu.BaseCompose:
    """
    Creating final transformation with normalization and
    casting to torch Tensor
    """
    return albu.Compose([
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            always_apply=True
        ),
        ToTensorV2()
    ])
