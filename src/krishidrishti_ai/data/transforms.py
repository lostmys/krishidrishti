from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, training: bool, robust: bool = False) -> transforms.Compose:
    steps: list[object] = [transforms.Resize((image_size, image_size))]
    if training:
        if robust:
            steps.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
            ])
        else:
            steps.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ])
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if training and robust:
        steps.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value=0))
    return transforms.Compose(steps)
