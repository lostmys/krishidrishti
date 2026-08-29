from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, training: bool) -> transforms.Compose:
    steps: list[object] = [transforms.Resize((image_size, image_size))]
    if training:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)
