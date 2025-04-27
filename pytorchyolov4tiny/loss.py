import os
import math
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import re
import time
import matplotlib.pyplot as plt

def to_cpu(tensor):
    return tensor.cpu()

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # box1: (N, 4), box2: (N, 4), where N is the number of boxes
    assert box1.size(0) == box2.size(0), "Number of boxes in box1 and box2 must match"

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:  # Transform from xywh to xyxy
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # Convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # Convex height
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # Convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # Center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU
            c_area = cw * ch + eps  # Convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def build_targets(p, targets, model):
    nt = targets.shape[0]  # Number of targets
    tcls, tbox, indices, anch, gt_boxes_pixel = [], [], [], [], []
    gain = torch.ones(7, device=targets.device)

    img_size = model.img_size  # e.g., 832

    for i, yolo_layer in enumerate(model.yolo_layers):
        na = yolo_layer.num_anchors  # Dynamically get number of anchors
        anchors = yolo_layer.anchors / yolo_layer.stride
        grid_size = model.img_size // yolo_layer.stride
        gain[2:6] = torch.tensor([grid_size, grid_size, grid_size, grid_size], device=targets.device)

        # Repeat targets for each anchor
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets_expanded = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        t = targets_expanded * gain
        #print("Anchors:", anchors)
        #print("Ground truth wh:", t[:, 4:6])

        if nt:
            r = t[:, :, 4:6] / anchors[:, None]
            j = torch.max(r, 1. / r).max(2)[0] < 4
            t = t[j]
        else:
            t = targets_expanded[0]

        b, c = t[:, :2].long().T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        gij = gxy.long()
        gi, gj = gij.T
        a = t[:, 6].long()

        # Calculate target box in pixel coordinates
        gt_xy_pixel = t[:, 2:4] * (img_size / grid_size)
        gt_wh_pixel = t[:, 4:6] * (img_size / grid_size)
        gt_boxes_pixel.append(torch.cat((gt_xy_pixel, gt_wh_pixel), 1))

        indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(anchors[a])
        tcls.append(c)

    return tcls, tbox, indices, anch, gt_boxes_pixel

def process_targets(targets, device):
    """Convert list of target tensors to a single tensor with image indices."""
    target_list = []
    for img_idx, target in enumerate(targets):
        if target.size(0) > 0:  # Check if there are boxes for this image
            target = target.to(device)
            img_indices = torch.full((target.size(0), 1), img_idx, dtype=torch.float32, device=device)
            target_with_idx = torch.cat((img_indices, target), dim=1)
            target_list.append(target_with_idx)
    if target_list:
        targets_tensor = torch.cat(target_list, dim=0)
    else:
        targets_tensor = torch.zeros(0, 6, device=device)  # Empty tensor if no targets
    return targets_tensor

def compute_loss(predictions, targets, model):
    device = targets.device
    lcls = torch.zeros(1, device=device)
    lbox = torch.zeros(1, device=device)
    lobj = torch.zeros(1, device=device)
    tcls, tbox, indices, anchors, gt_boxes_pixel = build_targets(predictions, targets, model)
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    total_iou_sum = torch.zeros(1, device=device)
    total_num_targets = 0
    for layer_index, layer_predictions in enumerate(predictions):
        yolo_layer = model.yolo_layers[layer_index]
        num_anchors = yolo_layer.num_anchors
        bbox_attrs = yolo_layer.bbox_attrs
        stride = yolo_layer.stride
        bs, channels, height, width = layer_predictions.shape
        assert channels == num_anchors * bbox_attrs
        layer_predictions = layer_predictions.view(bs, num_anchors, bbox_attrs, height, width)
        layer_predictions = layer_predictions.permute(0, 1, 3, 4, 2).contiguous()
        b, anchor, grid_j, grid_i = indices[layer_index]
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)
        num_targets = b.shape[0]
        if num_targets:
            ps = layer_predictions[b, anchor, grid_j, grid_i]
            pxy = ps[:, :2].sigmoid()
            if yolo_layer.new_coords:
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2
            else:
                pxy = ps[:, :2].sigmoid()
                pwh = torch.exp(ps[:, 2:4])
            pwh = pwh * anchors[layer_index]
            # Fix: Scale pwh by stride to convert to pixel units
            p_center_x = (pxy[:, 0] + grid_i.float()) * stride
            p_center_y = (pxy[:, 1] + grid_j.float()) * stride
            p_box_pixel = torch.cat((p_center_x.unsqueeze(1), p_center_y.unsqueeze(1), pwh * stride), dim=1)
            p_x1 = p_box_pixel[:, 0] - p_box_pixel[:, 2] / 2
            p_y1 = p_box_pixel[:, 1] - p_box_pixel[:, 3] / 2
            p_x2 = p_box_pixel[:, 0] + p_box_pixel[:, 2] / 2
            p_y2 = p_box_pixel[:, 1] + p_box_pixel[:, 3] / 2
            p_box_xyxy = torch.stack((p_x1, p_y1, p_x2, p_y2), dim=1)
            gt_box_pixel = gt_boxes_pixel[layer_index]
            gt_x1 = gt_box_pixel[:, 0] - gt_box_pixel[:, 2] / 2
            gt_y1 = gt_box_pixel[:, 1] - gt_box_pixel[:, 3] / 2
            gt_x2 = gt_box_pixel[:, 0] + gt_box_pixel[:, 2] / 2
            gt_y2 = gt_box_pixel[:, 1] + gt_box_pixel[:, 3] / 2
            gt_box_xyxy = torch.stack((gt_x1, gt_y1, gt_x2, gt_y2), dim=1)
            iou = bbox_iou(p_box_xyxy, gt_box_xyxy, x1y1x2y2=True, CIoU=True)
            #print(f"CIoU for layer {layer_index}: {iou.mean().item():.4f}")
            iou = iou.clamp(min=0)
            lbox += (1.0 - iou).mean()
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)
            total_iou_sum += iou.sum()
            total_num_targets += num_targets
            if ps.size(1) - 5 > 1:
                t = torch.zeros_like(ps[:, 5:], device=device)
                t[range(num_targets), tcls[layer_index]] = 1
                lcls += BCEcls(ps[:, 5:], t)
        lobj += BCEobj(layer_predictions[..., 4], tobj)
    lbox *= 1.0
    lobj *= 5.0
    lcls *= 1.5
    loss = lbox + lobj + lcls
    #print(f"lbox: {lbox.item():.4f}, lobj: {lobj.item():.4f}, lcls: {lcls.item():.4f}")
    if total_num_targets > 0:
        average_iou = total_iou_sum / total_num_targets
    else:
        average_iou = torch.zeros(1, device=device)
    metrics = torch.cat((lbox, lobj, lcls, loss, average_iou))
    return loss, to_cpu(metrics)

def yolo_loss(pred, targets, model, device):
    targets_tensor = process_targets(targets, device)
    loss, metrics = compute_loss(pred, targets_tensor, model)
    acc = metrics[4].item()
    return loss, acc