import torch
import numpy as np


def fit_ROI_in_frame(center, roi_size=72, load_size=256):
    center_w, center_h = center[0], center[1]
    center_h = roi_size // 2 if center_h < roi_size // 2 else center_h
    center_w = roi_size // 2 if center_w < roi_size // 2 else center_w
    center_h = load_size - roi_size // 2 if center_h > load_size - roi_size // 2 else center_h
    center_w = load_size - roi_size // 2 if center_w > load_size - roi_size // 2 else center_w
    return (center_w, center_h)


def crop_ROI(img, center, roi_size=72):
    return img[center[1] - roi_size // 2:center[1] + roi_size // 2,
               center[0] - roi_size // 2:center[0] + roi_size // 2]


def fit_ROI_in_frame_batch(centers, roi_size=72, load_size=256):
    return torch.tensor([fit_ROI_in_frame(center, roi_size, load_size) for center in centers])


def crop_ROI_batch(imgs, centers, roi_size=72):
    return torch.stack([
        img[:, center[1] - roi_size // 2:center[1] + roi_size // 2,
               center[0] - roi_size // 2:center[0] + roi_size // 2]
        for img, center in zip(imgs, centers)
    ], dim=0)