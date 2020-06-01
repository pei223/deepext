from typing import List, Iterable, Tuple
from torchvision.transforms import ToPILImage
import torch

import cv2
import numpy as np
from PIL import Image


def blend_result_and_source(source_img: Image.Image or np.ndarray, result_rgba_img_array: np.array) -> Image:
    """
    セグメンテーション結果と元画像を合成した画像を返す.
    :param source_img: 元画像
    :param result_rgba_img_array: 推論結果のRGBA画像
    :return: セグメンテーション結果と元画像を合成した画像
    """
    if isinstance(source_img, np.ndarray):
        source_img = Image.fromarray(source_img)
    source_img = source_img.convert("RGBA")
    result_mask = Image.new('RGBA', result_rgba_img_array.shape[:2], (255, 255, 255, 0))
    result_rgba_img = Image.fromarray(result_rgba_img_array).convert("RGBA")
    result_mask.paste(result_rgba_img, (0, 0), result_rgba_img)
    return Image.alpha_composite(source_img, result_rgba_img)


def transparent_only_black(img: Image.Image, alpha: float) -> Image.Image:
    result_img = img.convert("RGBA")
    newData = []
    for item in result_img.getdata():
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item[:3] + (alpha,))
    result_img.putdata(newData)
    return result_img


def draw_bounding_boxes_with_name_tag(origin_image: np.ndarray, bounding_boxes: Iterable[Iterable[float]], text: str,
                                      is_bbox_norm=False, thickness=1, color=(0, 0, 255)):
    image = origin_image.copy()
    height, width = image.shape[:2]
    for bounding_box in bounding_boxes:
        if len(bounding_box) < 4:
            continue
        x_min, y_min, x_max, y_max = bounding_box[:4]
        if is_bbox_norm:
            x_min *= width
            x_max *= width
            y_min *= height
            y_max *= height
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        image = draw_text_with_background(image, text=text, offsets=(x_min, y_min), background_color=color)
    return image


def draw_text_with_background(origin_img: np.ndarray, text, offsets: Tuple[int, int], background_color=(255, 255, 255),
                              text_color=(0, 0, 0), margin_px=5, font_scale=0.7):
    img = origin_img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    text_width, text_height = cv2.getTextSize(text, font, font_scale, 1)[0]
    background_coors = (offsets,
                        (int(offsets[0] + text_width + margin_px * 2), int(offsets[1] - text_height - margin_px * 2)))
    img = cv2.rectangle(img, background_coors[0], background_coors[1], background_color, cv2.FILLED)
    img = cv2.putText(img, text, (offsets[0] + margin_px, offsets[1] - margin_px), font, font_scale, text_color, 1,
                      cv2.LINE_AA)
    return img


def default_color_palette() -> List[int]:
    return [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0,
            192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64,
            0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64,
            192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128,
            64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64,
            192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128,
            192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192,
            192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128,
            0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96,
            0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64,
            128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128,
            224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192,
            160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224,
            0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160,
            64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224,
            64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32,
            128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128,
            64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224,
            128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128,
            192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128,
            160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192,
            160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224,
            192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224,
            192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128,
            96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32,
            96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96,
            0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64,
            160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64,
            224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64,
            160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64,
            224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]


def pil_to_cv(image: Image.Image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        return new_image
    elif new_image.shape[2] == 3:  # カラー
        return cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        return cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)


def cv_to_pil(image: np.ndarray):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        return Image.fromarray(new_image)
    elif new_image.shape[2] == 3:  # カラー
        return Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    elif new_image.shape[2] == 4:  # 透過
        return Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA))
    assert False, f"Invalid shape {new_image.shape}"


def img_to_pil(img: Image.Image or torch.Tensor or np.ndarray):
    if isinstance(img, torch.Tensor):
        return ToPILImage()(img)
    elif isinstance(img, np.ndarray):
        return cv_to_pil(img)
    else:
        return img


def img_to_cv2(img: Image.Image or torch.Tensor or np.ndarray):
    if isinstance(img, torch.Tensor):
        return (img.permute(1, 2, 0).detach().numpy() * 255).astype('uint32')
    if isinstance(img, Image.Image):
        return pil_to_cv(img)
    return img


def resize_image(img: Image.Image or torch.Tensor or np.ndarray, size: Tuple[int, int]):
    """
    NOTE Tensorオブジェクトならリサイズ不能のためそのまま返す.
    :param img:
    :param size:
    """
    if isinstance(img, Image.Image):
        return img.resize(size)
    elif isinstance(img, np.ndarray):
        return cv2.resize(img, size)
    print("Resizing tensor is not enable.")
    return img


def get_image_size(img: Image.Image or torch.Tensor or np.ndarray):
    if isinstance(img, Image.Image):
        return img.size
    elif isinstance(img, np.ndarray):
        return img.shape[:2]
    return img.shape[1:]
