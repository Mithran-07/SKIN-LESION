"""
Albumentations-based augmentation pipelines for dermoscopic images.

Key dermoscopy-specific augmentations:
- HairAugmentation: Synthesizes realistic body hair occlusion artifacts
- ElasticTransform: Simulates slight biological deformation
- CoarseDropout: Simulates bubbles and occlusion artifacts
"""

import random
from typing import Dict

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HairAugmentation(A.ImageOnlyTransform):
    """
    Custom Albumentations transform that overlays synthetic hair strands.

    Body hair is one of the most pervasive artifacts in dermoscopy,
    causing networks to learn spurious correlations if not regularized.
    This transform synthesizes Bezier-curve hair strands during training.
    """

    def __init__(self, num_hairs: int = 10, p: float = 0.5):
        super().__init__(p=p)
        self.num_hairs = num_hairs

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        for _ in range(random.randint(1, self.num_hairs)):
            # Random Bezier control points
            x0, y0 = random.randint(0, w), random.randint(0, h)
            x1, y1 = x0 + random.randint(-80, 80), y0 + random.randint(-80, 80)
            x2, y2 = x1 + random.randint(-80, 80), y1 + random.randint(-80, 80)

            # Draw a thin dark curve simulating hair
            pts = np.array(
                [
                    [int(((1 - t) ** 2) * x0 + 2 * (1 - t) * t * x1 + t ** 2 * x2),
                     int(((1 - t) ** 2) * y0 + 2 * (1 - t) * t * y1 + t ** 2 * y2)]
                    for t in np.linspace(0, 1, 50)
                ],
                dtype=np.int32,
            )
            color = random.choice([
                (20, 15, 10),   # Dark brown
                (10, 8, 5),     # Near-black
                (80, 60, 40),   # Medium brown
            ])
            thickness = random.randint(1, 2)
            for i in range(len(pts) - 1):
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)
        return img

    def get_transform_init_args_names(self):
        return ("num_hairs",)


def get_train_transforms(aug_cfg: Dict) -> A.Compose:
    """
    Build the full training augmentation pipeline.

    Applies spatial, color, and dermoscopy-specific augmentations.
    All operations are done in Albumentations for performance.
    """
    crop_cfg = aug_cfg.get("random_resized_crop", {})
    norm_cfg = aug_cfg.get("normalize", {})

    transforms = [
        A.RandomResizedCrop(
            height=crop_cfg.get("height", 224),
            width=crop_cfg.get("width", 224),
            scale=tuple(crop_cfg.get("scale", [0.8, 1.0])),
            p=1.0,
        ),
        A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
        A.VerticalFlip(p=aug_cfg.get("vertical_flip_p", 0.5)),
        A.Rotate(limit=30, p=0.4),
        A.ColorJitter(
            brightness=aug_cfg.get("color_jitter", {}).get("brightness", 0.2),
            contrast=aug_cfg.get("color_jitter", {}).get("contrast", 0.2),
            saturation=aug_cfg.get("color_jitter", {}).get("saturation", 0.2),
            hue=aug_cfg.get("color_jitter", {}).get("hue", 0.1),
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=1, sigma=50, p=aug_cfg.get("elastic_transform_p", 0.3)
        ),
        A.CoarseDropout(
            max_holes=8, max_height=16, max_width=16,
            fill_value=0, p=aug_cfg.get("coarse_dropout_p", 0.2)
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),  # Enhance local contrast (dermoscopy-relevant)
    ]

    if aug_cfg.get("hair_augmentation", True):
        transforms.append(HairAugmentation(num_hairs=15, p=0.5))

    transforms += [
        A.Normalize(
            mean=norm_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=norm_cfg.get("std", [0.229, 0.224, 0.225]),
        ),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms(aug_cfg: Dict) -> A.Compose:
    """
    Build validation / inference augmentation pipeline.

    Only applies deterministic resizing and normalization.
    """
    norm_cfg = aug_cfg.get("normalize", {})
    img_size = aug_cfg.get("random_resized_crop", {}).get("height", 224)

    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(
            mean=norm_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=norm_cfg.get("std", [0.229, 0.224, 0.225]),
        ),
        ToTensorV2(),
    ])
