"""
Augmentation Pipelines — Phase 2
Albumentations transforms for training, validation, and test sets.
Includes custom HairAugmentation for dermoscopy-specific simulation.
"""

import random
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────
#  Custom Hair Augmentation (Dermoscopy-specific)
# ─────────────────────────────────────────────────────────
class HairAugmentation(A.ImageOnlyTransform):
    """
    Simulates hair artifacts common in dermoscopy images.
    Draws random Bezier-like curved lines to mimic dark/light hairs.
    """

    def __init__(
        self,
        num_hairs: tuple = (3, 10),
        hair_width: tuple = (1, 3),
        dark_prob: float = 0.7,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_hairs   = num_hairs
        self.hair_width  = hair_width
        self.dark_prob   = dark_prob

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        h, w = img.shape[:2]
        n = random.randint(*self.num_hairs)

        for _ in range(n):
            # Random Bezier control points
            x0, y0 = random.randint(0, w), random.randint(0, h)
            x1, y1 = random.randint(0, w), random.randint(0, h)
            xc, yc = random.randint(0, w), random.randint(0, h)  # control point

            thickness = random.randint(*self.hair_width)
            is_dark   = random.random() < self.dark_prob
            color     = (10, 10, 10) if is_dark else (220, 220, 180)

            # Draw quadratic Bezier curve as polyline
            pts = []
            for t in np.linspace(0, 1, 30):
                px = int((1 - t)**2 * x0 + 2*(1-t)*t * xc + t**2 * x1)
                py = int((1 - t)**2 * y0 + 2*(1-t)*t * yc + t**2 * y1)
                pts.append([px, py])

            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=False,
                          color=color, thickness=thickness,
                          lineType=cv2.LINE_AA)

        return img

    def get_transform_init_args_names(self):
        return ("num_hairs", "hair_width", "dark_prob")


# ─────────────────────────────────────────────────────────
#  Augmentation Pipelines
# ─────────────────────────────────────────────────────────
def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Heavy augmentation pipeline for training."""
    return A.Compose([
        # ── Spatial
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.6),

        # ── Elastic deformation
        A.ElasticTransform(
            alpha=60,
            sigma=6,
            p=0.3
        ),

        # ── Color / Intensity
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.6
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4
        ),

        # ── Dermoscopy-specific
        HairAugmentation(num_hairs=(2, 8), p=0.4),

        # ── Coarse Dropout (hide patches)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 40),
            hole_width_range=(16, 40),
            fill=0,
            p=0.3
        ),

        # ── Normalize + Tensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Deterministic pipeline for validation/test sets."""
    return A.Compose([
        A.Resize(int(image_size * 256 / 224), int(image_size * 256 / 224)),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_test_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Same as validation — no stochastic ops."""
    return get_val_transforms(image_size)


# ─────────────────────────────────────────────────────────
#  Augmentation Preview
# ─────────────────────────────────────────────────────────
def generate_augmentation_preview(img_path: str, output_path: str, n: int = 8):
    """
    Generate a grid showing n augmented versions of a single image.
    Useful for verifying augmentation quality.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open(img_path).convert("RGB"))
    transform = get_train_transforms()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Augmentation Preview", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        aug_img = transform(image=img)["image"]
        # Denormalize for display
        mean = np.array(IMAGENET_MEAN)
        std  = np.array(IMAGENET_STD)
        display = aug_img.permute(1, 2, 0).numpy()
        display = (display * std + mean).clip(0, 1)
        ax.imshow(display)
        ax.set_title(f"Aug #{i+1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK   ] Augmentation preview -> {output_path}")
