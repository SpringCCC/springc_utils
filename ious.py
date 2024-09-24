import torch
import math
from .data_convert import *

def CIoU(boxa, boxb):
    """
    # 当boxes与真实box重合时，且都在在中心点重合时，一个长宽比接近真实box，一个差异很大
    # 我们认为长宽比接近的是比较好的，损失应该是比较小的。所以ciou增加了对box长宽比的考虑
    ciou=iou-两个box中心点距离平方/最小闭包区域对角线距离平方-alpha*v
    loss=1-iou+两个box中心点距离平方/最小闭包区域对角线距离平方+alpha*v
    注意loss跟上边不一样，这里不是1-ciou
    v用来度量长宽比的相似性,4/(pi *pi)*(arctan(boxa_w/boxa_h)-arctan(boxb_w/boxb_h))^2
    alpha是权重值，衡量ciou公式中第二项和第三项的权重，
    alpha大优先考虑v，alpha小优先考虑第二项距离比,alpha = v / ((1 - iou) + v)。
    """
    # 求交集
    inter_x1, inter_y1 = torch.maximum(boxa[:, 0], boxb[:, 0]), torch.maximum(boxa[:, 1], boxb[:, 1])
    inter_x2, inter_y2 = torch.minimum(boxa[:, 2], boxb[:, 2]), torch.minimum(boxa[:, 3], boxb[:, 3])
    inter_h = torch.maximum(torch.tensor([0]), inter_y2 - inter_y1)
    inter_w = torch.maximum(torch.tensor([0]), inter_x2 - inter_x1)
    inter_area = inter_w * inter_h

    # 求并集
    union_area = ((boxa[:, 3] - boxa[:, 1]) * (boxa[:, 2] - boxa[:, 0])) + \
                 ((boxb[:, 3] - boxb[:, 1]) * (boxb[:, 2] - boxb[:, 0])) - inter_area + 1e-8  # + 1e-8 防止除零

    # 求最小闭包区域的x1,y1,x2,y2
    ac_x1, ac_y1 = torch.minimum(boxa[:, 0], boxb[:, 0]), torch.minimum(boxa[:, 1], boxb[:, 1])
    ac_x2, ac_y2 = torch.maximum(boxa[:, 2], boxb[:, 2]), torch.maximum(boxa[:, 3], boxb[:, 3])

    # 把两个bbox的x1,y1,x2,y2转换成ctr_x,ctr_y
    boxa_ctrx, boxa_ctry = boxa[:, 0] + (boxa[:, 2] - boxa[:, 0]) / 2, boxa[:, 1] + (boxa[:, 3] - boxa[:, 1]) / 2
    boxb_ctrx, boxb_ctry = boxb[:, 0] + (boxb[:, 2] - boxb[:, 0]) / 2, boxb[:, 1] + (boxb[:, 3] - boxb[:, 1]) / 2
    boxa_w, boxa_h = boxa[:, 2] - boxa[:, 0], boxa[:, 3] - boxa[:, 1]
    boxb_w, boxb_h = boxb[:, 2] - boxb[:, 0], boxb[:, 3] - boxb[:, 1]

    # 求两个box中心点距离平方length_box_ctr，最小闭包区域对角线距离平方length_ac
    length_box_ctr = (boxb_ctrx - boxa_ctrx) * (boxb_ctrx - boxa_ctrx) + \
                     (boxb_ctry - boxa_ctry) * (boxb_ctry - boxa_ctry)
    length_ac = (ac_x2 - ac_x1) * (ac_x2 - ac_x1) + (ac_y2 - ac_y1) * (ac_y2 - ac_y1)

    v = (4 / (math.pi * math.pi)) * (torch.atan(boxa_w / boxa_h) - torch.atan(boxb_w / boxb_h)) \
        * (torch.atan(boxa_w / boxa_h) - torch.atan(boxb_w / boxb_h))
    iou = inter_area / (union_area + 1e-8)
    alpha = v / ((1 - iou) + v)
    ciou = iou - length_box_ctr / length_ac - alpha * v
    # ciou_loss = 1 - iou + length_box_ctr / length_ac + alpha * v
    return ciou

def bbox1_bbox2_cious(bbox1, bbox2, normalize=False):
    # bbox1: M, 4
    # bbox2: N, 4
    bbox1, bbox2 = toTensor(bbox1), toTensor(bbox2)
    m, n = len(bbox1), len(bbox2)
    cious = torch.zeros(m, n)
    bbox1, bbox2 = toTensor(bbox1), toTensor(bbox2)
    for i, box1 in enumerate(bbox1):
        rbox1 = box1.reshape(-1, 4).repeat(n, 1)
        cious[i] = CIoU(rbox1, bbox2)
    return cious if not normalize else (cious + 1) / 2

def bbox1_bbox2_ious(box1, box2):
    # bbox1: M, 4
    # bbox2: N, 4
    box1, box2 = toTensor(box1), toTensor(box2)
    bbox1 = box1[:,None] #M, 1, 4
    bbox2 = box2[None] #1, N, 4
    tl = torch.maximum(bbox1[:, :, :2], bbox2[:, :, :2])
    br = torch.minimum(bbox1[:, :, 2:], bbox2[:, :, 2:])
    inter = torch.prod(br-tl, dim=-1) * torch.all(br>tl, dim=-1)
    area1 = torch.prod(box1[:, 2:] - box1[:, :2], dim=-1)
    area2 = torch.prod(box2[:, 2:] - box2[:, :2], dim=-1)
    iou = inter / (area1[:,None] + area2[None] - inter)
    return iou