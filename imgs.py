import cv2
import numpy as np
import random
import numpy as np
import cv2
from .data_convert import toNumpy
import random
from PIL import Image

def read_img(img_path):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

def save_img(img, sv_path, suffix=".jpg"):
    cv2.imencode(suffix, img)[1].tofile(sv_path)



def resize_pad_image(image, size, pad_v=128, is_center=True, is_mod=False, stride=32):
    """
        对输入图像进行等比例resize，然后缩放到指定尺寸size,不足的部分进行pad
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


class RandomHSV:

    """
    HSV 色域变换
    hgain（色相增益）：
        影响图像的色相（Hue），即颜色的基本属性（如红色、绿色等）。
        增益值范围为 -1 到 1，正值会使色相偏向其原本的颜色，负值则可能导致颜色的变化，例如将红色偏向蓝色。

    sgain（饱和度增益）：
        控制图像的饱和度（Saturation），即颜色的鲜艳程度。
        增益值大于 1 会增加饱和度，使颜色更鲜艳；小于 1 则会降低饱和度，使颜色更接近灰色。

    vgain（明度增益）：
        影响图像的明度（Value），即亮度的程度。
        增益值大于 1 会使图像变亮，小于 1 则会使图像变暗。
    h, s, v=1时，相当于不做任何色域变换
    """
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img):
        # 输入img：opencv读取的BGR通道图像
        """Applies random horizontal or vertical flip to an image with a given probability."""
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BRG2HSV))
        dtype = img.dtype  # uint8
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BRG)  # no return needed
        return img