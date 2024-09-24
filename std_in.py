import numpy as np
import cv2
from .data_convert import toNumpy
import random
from PIL import Image


def prepare_inference_input(x, size, m=0, std=255, is_center=True, pad_v=128, stride=32, is_mod=False):
    """
    x: cv2读取的原始图像，也就是BGR格式， 0-255
    size: (h, w) 期待的h和w

    返回： (1, 3, h, w)  0-1
    """
    x = x[:, :, ::-1] #bgr转成rgb
    x = resize_pad_image(x, size, pad_v, is_center, is_mod, stride)
    x = (x - m) / std
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x


def sc_rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def distor_image(image, size, jitter=0.3):
    """
    扭曲图像，为了mosaic做准备
    """
    if isinstance(size, tuple) or isinstance(size, list):
        size = size[0] 
    iw, ih = image.size
    flip = np.random.rand()<0.5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #   对图像进行缩放并且进行长和宽的扭曲
    new_ar = iw/ih * sc_rand(1-jitter,1+jitter) / sc_rand(1-jitter,1+jitter)
    scale = sc_rand(.4, 1)
    if new_ar < 1:
        nh = int(scale * size)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * size)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)
    return image, flip

def color_space_convert(image, hue=.1, sat=0.7, val=0.4):
    #   对图像进行色域变换
    #   计算色域变换的参数
    r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    hue, sat, val   = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    dtype           = image.dtype
    #   应用变换
    x       = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
    return new_image



def resize_image(image, size):
    """
        对输入图像进行等比例resize，然后缩放到指定尺寸size,
    Args:
        image：opencv读取的图像，np.ndarray格式
        size:目标尺寸, (h, w)
    Returns:指定尺寸的图像

    """
    if isinstance(size, tuple):
        size = size[0]
    image = toNumpy(image)
    ih, iw, _ = image.shape
    h, w = size
    r = min(w / iw, h / ih)
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if r>1 else cv2.INTER_AREA
        image = cv2.resize(image, (min(math.ceil(iw * r), size), min(math.ceil(ih * r), size)), interpolation=interp)
    return image


def resize_pad_image(image, size, pad_v=128, is_center=True, is_mod=False, stride=32):
    """
        对输入图像进行等比例resize，然后缩放到指定尺寸size,
    Args:
        image：opencv读取的图像，np.ndarray格式
        size:目标尺寸, (h, w)
        is_center:是否居中放置
        is_mod, stride：合并起来使用，is_mod主要是在推理时缩小图像pad的范围，但同时也能满足stride的需求
            比如：一个图像，等比例resize后为（640， 309），
                如果is_mode为False，那么309需要pad到640
                如果is_mod为True，那么309只需要pad到320，
                这样可以减小图像的size，减少推理时间
    Returns:指定尺寸的图像

    """
    if isinstance(size, int):
        size = (size, size)
    image = toNumpy(image)
    ih, iw, _ = image.shape
    h, w = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    # 缩小图像：一般使用 cv2.INTER_AREA 可以得到较好的效果。
    # 放大图像：一般使用 cv2.INTER_CUBIC 或 cv2.INTER_LANCZOS4 可以得到更好的效果
    image = cv2.resize(image, (nw, nh), interpolation= cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    dh, dw = h - nh, w - nw
    if is_mod:
        dh, dw = np.mod(dh, stride), np.mod(dw, stride)  # wh padding
    if is_center:
        top, left = dh//2, dw//2
    else:
        top = random.randint(0, dh)
        left = random.randint(0, dw)
    bottom, right = dh - top, dw - left
    img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_v, pad_v, pad_v))  # add border
    return img