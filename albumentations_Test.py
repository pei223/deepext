import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor
from PIL import Image
import numpy as np
import cv2
import torch

transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(),
    A.Rotate((-10, 10)),
    ToTensorV2(),
], bbox_params=A.BboxParams(label_fields=['category_ids'], format="pascal_voc"))

d = {
    0: "aaa",
    1: "hoge",
    2: "aaaaa"
}
image = np.array(Image.open("D:/dataset/segmentation/test_seg/dataset/images/IMG_6531.png"))
bboxes = np.array([[100, 100, 300, 300, 0], [100, 100, 500, 500, 1], [100, 100, 150, 300,3]])


result = transform(image=image, bboxes=bboxes[:, :4], category_ids=bboxes[:, 4])
image = result["image"]
bbox = result["bboxes"]
c_ids = result["category_ids"]

print(image)
print(bbox)
print(c_ids)
