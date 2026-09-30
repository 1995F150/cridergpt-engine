import torch
from PIL import Image

from generate import tensor_to_pil


def test_tensor_to_pil():
    x = torch.zeros(1, 3, 256, 256)
    image = tensor_to_pil(x)
    assert isinstance(image, Image.Image)
    assert image.size == (256, 256)
    assert image.mode == "RGB"


def test_tensor_to_pil_clamps():
    x = torch.full((1, 3, 8, 8), 5.0)
    image = tensor_to_pil(x)
    assert image.getpixel((0, 0)) == (255, 255, 255)
