import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


def try_cuda(e):
    if torch.cuda.is_available() and hasattr(e, "cuda"):
        return e.cuda()
    return e


def parameter_count(model: torch.nn.Module):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params


def cv2image_to_tensor(cv2image: np.ndarray, norm=True):
    img_array = cv2image.transpose(2, 0, 1)
    if norm:
        img_array = img_array / 255
    return torch.FloatTensor(torch.from_numpy(img_array.astype('float32')))


def img_to_tensor(img: Image.Image or torch.Tensor or np.ndarray):
    if isinstance(img, Image.Image):
        return ToTensor()(img)
    if isinstance(img, np.ndarray):
        return cv2image_to_tensor(img)
    return img
