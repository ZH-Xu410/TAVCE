import os
from src.facerender.model import AnimateModel


def load_animate_model(cfg, device, is_train=True, is_cmtc=False):
    animate_model = AnimateModel(cfg, device=device, is_train=is_train, is_cmtc=is_cmtc)
    pretrain_weight = cfg.MODEL.get('CHECKPOINT', '')
    if os.path.exists(pretrain_weight):
        animate_model.load(pretrain_weight)
        print('load checkpoint from', pretrain_weight)

    return animate_model
