import os
import torch
from src.audio2exp_models.audio2exp import Audio2Exp
from src.audio2exp_models.networks import SimpleWrapperV2


def load_audio2exp_model(cfg, device, is_train=True):
    netG = SimpleWrapperV2().to(device)
    audio2exp_model = Audio2Exp(netG, cfg, device=device, is_train=is_train)
    pretrain_weight = cfg.MODEL.get('CHECKPOINT', '')
    if os.path.exists(pretrain_weight):
        audio2exp_model.load(pretrain_weight)
        print('load checkpoint from', pretrain_weight)
    return audio2exp_model
