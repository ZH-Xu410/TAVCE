import torch
from torch import nn
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from src.audio2pose_models.cvae import CVAE
from src.audio2pose_models.discriminator import PoseSequenceDiscriminator
from src.audio2pose_models.audio_encoder import AudioEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class Audio2Pose(nn.Module):
    def __init__(self, cfg, wav2lip_checkpoint='checkpoints/wav2lip.pth', device="cuda", **kwargs):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.MODEL.CVAE.SEQ_LEN
        self.latent_dim = cfg.MODEL.CVAE.LATENT_SIZE
        self.device = device

        self.audio_encoder = AudioEncoder(wav2lip_checkpoint).to(device)

        self.netG = CVAE(cfg).to(device)

    def forward(self, x):
        result = {}
        coeff_gt = x["coeff_gt"]  # bs frame_len 73
        result["pose_motion_gt"] = (
            coeff_gt[:, :, 64:70] - coeff_gt[:, :1, 64:70]
        )  # bs frame_len 6
        result["ref"] = coeff_gt[:, 0, 64:70]  # bs  6
        result["class"] = x["class"].view(-1)  # bs
        indiv_mels = x["indiv_mels"]  # bs frame_len 1 80 16

        # forward
        audio_emb = self.audio_encoder(indiv_mels)  # bs frame_len 512
        result["audio_emb"] = audio_emb
        result = self.netG(result)

        pose_motion_pred = result["pose_motion_pred"]  # bs frame_len 6
        pose_gt = coeff_gt[:, :, 64:70].clone()  # bs frame_len 6
        pose_pred = coeff_gt[:, :1, 64:70] + pose_motion_pred  # bs frame_len 6

        result["pose_pred"] = pose_pred
        result["pose_gt"] = pose_gt

        return result

    def test(self, x):
        result = {}

        indiv_mels = x["indiv_mels"]  # bs T 1 80 16
        num_frames = int(x["num_frames"]) - 1


        ref = x["ref"]  # 1 1 70
        result["ref"] = ref[:1, 0, 64:70].repeat(indiv_mels.shape[0], 1)
        result["class"] = x["class"]
        bs = ref.shape[0]

        #
        div = num_frames // self.seq_len
        re = num_frames % self.seq_len
        pose_motion_pred_list = [
            torch.zeros(
                result["ref"].unsqueeze(1).shape,
                dtype=result["ref"].dtype,
                device=result["ref"].device,
            )
        ]

        for i in tqdm(range(div), "audio2pose"):
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            result["z"] = z
            audio_emb = self.audio_encoder(
                indiv_mels[:, i *
                           self.seq_len: (i + 1) * self.seq_len, :, :, :]
            )  # bs seq_len 512
            result["audio_emb"] = audio_emb
            result = self.netG.test(result)
            pose_motion_pred_list.append(
                result["pose_motion_pred"]
            )  # list of bs seq_len 6

        if re != 0:
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            result["z"] = z
            audio_emb = self.audio_encoder(
                indiv_mels[:, -1 * self.seq_len:, :, :, :]
            )  # bs seq_len  512
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len - audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1)
                audio_emb = torch.cat([pad_audio_emb, audio_emb], 1)
            result["audio_emb"] = audio_emb
            result = self.netG.test(result)
            pose_motion_pred_list.append(
                result["pose_motion_pred"][:, -1 * re:, :])

        pose_motion_pred = torch.cat(pose_motion_pred_list, dim=1)
        result["pose_motion_pred"] = pose_motion_pred

        pose_pred = ref[:1, :1, 64:70] + pose_motion_pred  # bs T 6

        result["pose_pred"] = pose_pred
        return result

    def load(self, ckpt):
        ckpt = torch.load(ckpt, self.device)
        self.netG.load_state_dict(ckpt['netG'])
        self.audio_encoder.load_state_dict(ckpt['audio_encoder'])
        if hasattr(self, 'netD'):
            self.netD.load_state_dict(ckpt['netD'])
