import albumentations as A
import numpy as np
import cv2


class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']
    
def get_transforms(model_name, seed=0):
    if model_name == "inception_v3":
        img_size = 299
    else:
        img_size = 224

    train_transforms = Transforms(
        A.Compose([
            A.Resize(img_size, img_size),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.8
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(1, 20), hole_width_range=(1, 20), fill=0, p=0.3),
            A.RandomShadow(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            A.ToTensorV2()
        ], seed=seed),
    )
    
    test_transforms = Transforms(
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            A.ToTensorV2()
        ], seed=seed)
    )
    return train_transforms, test_transforms