import os
import torch
from src.audio2pose_models.audio2pose import Audio2Pose


def load_audio2pose_model(cfg, device, is_train=True, is_cmtc=False):
    audio2pose_model = Audio2Pose(cfg, 'checkpoints/wav2lip.pth', device=device, is_train=is_train, is_cmtc=is_cmtc)
    pretrain_weight = cfg.MODEL.get('CHECKPOINT', '')
    if os.path.exists(pretrain_weight):
        audio2pose_model.load(pretrain_weight)
        print('load checkpoint from', pretrain_weight)
    return audio2pose_model
