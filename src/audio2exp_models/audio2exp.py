from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from src.face3d.models.bfm import ParametricFaceModel
from src.face3d.util.nvdiffrast import MeshRenderer
from src.lip_reading import load_model, crop_mouth_area, get_preprocessing_pipelines


class Audio2Exp(nn.Module):
    def __init__(self, netG, cfg, device, **kwargs):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.device = device
        self.netG = netG.to(device)

    def forward(self, batch):
        mel_input = batch["indiv_mels"]  # bs T 1 80 16
        ref = batch["ref"][:, :, :64]  # bs T 64
        ratio = batch["ratio_gt"]  # bs T 1

        audiox = mel_input.view(-1, 1, 80, 16)  # bs*T 1 80 16

        exp_coeff_pred = self.netG(audiox, ref, ratio)  # bs T 64

        # BS x T x 64
        results_dict = {"exp_coeff_pred": exp_coeff_pred}
        return results_dict

    def test(self, batch, seq_len=10):
        mel_input = batch["indiv_mels"]  # bs T 1 80 16
        num_frames = mel_input.shape[1]

        exp_coeff_pred = []

        div = num_frames // seq_len
        re = num_frames % seq_len

        ref = batch["ref"][:1, :1, :64].repeat(mel_input.shape[0], seq_len, 1)
        for i in tqdm(range(div), "audio2exp"):  # every 10 frames
            current_mel_input = mel_input[:, i*seq_len:(i+1)*seq_len]

            # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            
            ratio = batch["ratio_gt"][:, i*seq_len:(i+1)*seq_len]  # bs T

            audiox = current_mel_input.view(-1, 1, 80, 16)  # bs*T 1 80 16

            curr_exp_coeff_pred = self.netG(audiox, ref, ratio)  # bs T 64

            exp_coeff_pred += [curr_exp_coeff_pred]
        
        if re != 0:
            current_mel_input = mel_input[:, -seq_len:]

            # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ratio = batch["ratio_gt"][:, -seq_len:]  # bs T

            audiox = current_mel_input.view(-1, 1, 80, 16)  # bs*T 1 80 16

            curr_exp_coeff_pred = self.netG(audiox, ref, ratio)  # bs T 64

            exp_coeff_pred += [curr_exp_coeff_pred[:, -re:]]

        # BS x T x 64
        results_dict = {"exp_coeff_pred": torch.cat(exp_coeff_pred, axis=1)}
        return results_dict
    
    def load(self, ckpt):
        ckpt = torch.load(ckpt, self.device)
        self.netG.load_state_dict(ckpt['netG'])
