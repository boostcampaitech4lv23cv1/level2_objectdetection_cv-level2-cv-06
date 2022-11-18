import albumentations as A
from albumentations.core.composition import Compose


def soft_aug():
    return Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ]
    )


def hard_aug():
    return Compose(
        [
            A.Resize(224, 224),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=0.7),
                    A.MedianBlur(blur_limit=3, p=0.7),
                    A.GaussianBlur(blur_limit=3, p=0.7),
                    A.GaussNoise(var_limit=(3.0, 9.0), p=0.7),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=10,
                max_height=20,
                max_width=20,
                min_holes=1,
                min_height=3,
                min_width=3,
                p=0.3,
            ),
        ]
    )
