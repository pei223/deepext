import numpy as np
from PIL import Image
import albumentations as A
import cv2


class AlbumentationsClsWrapperTransform:
    def __init__(self, albumentations_transforms: A.Compose, is_image_normalize=True):
        self._albumentations_transforms = albumentations_transforms
        self._is_image_normalize = is_image_normalize

    def __call__(self, image: Image.Image or np.ndarray, label: int):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result_dict = self._albumentations_transforms(image=image)
        result_image = result_dict["image"]
        if self._is_image_normalize:
            result_image = result_image / 255.
        return result_image, label


class AlbumentationsOnlyImageWrapperTransform:
    def __init__(self, albumentations_transforms: A.Compose, is_image_normalize=True):
        self._albumentations_transforms = albumentations_transforms
        self._is_image_normalize = is_image_normalize

    def __call__(self, image: Image.Image or np.ndarray):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result_dict = self._albumentations_transforms(image=image)
        result_image = result_dict["image"]
        if self._is_image_normalize:
            result_image = result_image / 255.
        return result_image
