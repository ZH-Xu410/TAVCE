import torch
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from yacs.config import CfgNode as Config
from src.model_helper import load_model
from src.datasets.dataloader import get_test_dataloader
from src.utils.face_enhancer import enhancer_list
from correlation.models import AudioEncoder

import safetensors
import safetensors.torch
import face_alignment
from PIL import Image
from src.face3d.models import networks
from src.face3d.util.load_mats import load_lm3d
from preprocess.recon_3dmm import recon_3dmm
from src.utils.safetensor_helper import load_x_from_safetensor
from src.face3d.util.preprocess import align_img

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
torch.set_num_threads(cpu_num )


port = 12355

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def init_3dmm_recon():
        sadtalker_path = {
        "use_safetensor": True,
        "checkpoint": "checkpoints/SadTalker_V0.0.2_256.safetensors",
        "dir_of_BFM_fitting": "src/config",
        }
        net_recon = networks.define_net_recon(
        net_recon="resnet50", use_last_fc=False, init_path=""
        ).cuda()
        checkpoint = safetensors.torch.load_file(sadtalker_path["checkpoint"])
        net_recon.load_state_dict(load_x_from_safetensor(checkpoint, "face_3drecon"))
        net_recon.eval()
        net_recon = net_recon
        lm3d_std = load_lm3d(sadtalker_path["dir_of_BFM_fitting"])
        lm_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda', face_detector='sfd')
        return net_recon, lm3d_std, lm_predictor


def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80:144]
    tex_coeffs = coeffs[:, 144:224]
    angles = coeffs[:, 224:227]
    gammas = coeffs[:, 227:254]
    translations = coeffs[:, 254:]
    return {
        "id": id_coeffs,
        "exp": exp_coeffs,
        "tex": tex_coeffs,
        "angle": angles,
        "gamma": gammas,
        "trans": translations,
    }



def recon_3dmm(img, lm3d_std, net_recon, lm_predictor, device):
    frame = img.permute(1, 2, 0).cpu().numpy() * 255
    frame = frame.astype(np.uint8)

    lm = lm_predictor.get_landmarks_from_image(frame)[0]

    frame = Image.fromarray(frame)
    W, H = frame.size
    lm[:, -1] = H - 1 - lm[:, -1]

    trans_params, im, lm, _ = align_img(frame, lm, lm3d_std)
    trans_params = np.array(
        [float(item) for item in np.hsplit(trans_params, 5)]
    ).astype(np.float32)
    im_t = (
        torch.tensor(np.array(im) / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .to(device)
        .unsqueeze(0)
    )

    with torch.no_grad():
        full_coeff = net_recon(im_t)
        coeffs = split_coeff(full_coeff)

    pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}

    pred_coeff = np.concatenate(
        [
            pred_coeff["exp"],
            pred_coeff["angle"],
            pred_coeff["trans"],
            trans_params[2:][None],
        ],
        1,
    )
    
    return pred_coeff



def cov_matrix(x, y):
    """
    args:
      x: (B, C)
      y: (B, C)
    return:
      m: (B, C, C)
    """

    z = torch.stack([x, y], dim=1)
    m = z - z.mean(1, keepdim=True)
    m = torch.bmm(m.transpose(1, 2), m)
    return m


@torch.no_grad()
def main(rank, world_size, args):
    if world_size > 1:
        ddp_setup(rank, world_size)
    
    if rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)

    cfg = Config.load_cfg(open(args.config))
    audio2exp_cfg = Config.load_cfg(open(cfg['audio2exp_cfg']))
    audio2pose_cfg = Config.load_cfg(open(cfg['audio2pose_cfg']))
    facerender_cfg = Config.load_cfg(open(cfg['facerender_cfg']))
    audio2exp_cfg['MODEL']['CHECKPOINT'] = args.audio2exp_ckpt
    audio2pose_cfg['MODEL']['CHECKPOINT'] = args.audio2pose_ckpt
    facerender_cfg['MODEL']['CHECKPOINT'] = args.facerender_ckpt

    # create a dataset
    dataloader = get_test_dataloader(cfg.DATA, ddp=world_size > 1)

    device = 'cuda'

    # create a model
    audio2exp_model = load_model(audio2exp_cfg, device, is_train=False).eval()
    audio2pose_model = load_model(audio2pose_cfg, device, is_train=False).eval()
    facerender_model = load_model(facerender_cfg, device, is_train=False, is_cmtc=True).eval()
    audio_encoder = AudioEncoder().to(device).eval()
    ckpt = torch.load(cfg["cmtc_image_ckpt"], "cpu")
    audio_encoder.load_state_dict(
        {k.replace("module.", ""): v for k, v in ckpt["audio_encoder"].items()}
    )
    net_recon, lm3d_std, lm_predictor = init_3dmm_recon()


    for i, data in enumerate(dataloader):
        save_dir = os.path.join(args.result_dir, data['name'][0])
        if os.path.exists(save_dir):
            print(f'skip existing video {i+1}')
            continue

        print(f'processing video {i*world_size+1}')
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
            
        if 'ref' not in data:
            ref_coeff = recon_3dmm(data["source_image"][0, 0], lm3d_std=lm3d_std, net_recon=net_recon, lm_predictor=lm_predictor, device=device)
            data["ref"] = torch.from_numpy(ref_coeff).unsqueeze(0).cuda()
        
        T = data['source_image'].shape[1]
        audio_correlations = []
        for j in tqdm(range(T), "audio correlation"):
            audios = data["indiv_mels"][:, [max(j-1, 0), j]].contiguous()
            bs, _, ca, ha, wa = audios.shape
            audio_features_batch = audio_encoder(audios.view(bs * 2, ca, ha, wa)).view(bs, 2, -1)
            audio_correlations.append(cov_matrix(
                audio_features_batch[:, 0], audio_features_batch[:, 1] #, facerender_model.mu_a[0], facerender_model.mu_a[1]
            ))
        data['cor'] = torch.stack(audio_correlations, dim=1)
        exp_coeff_pred = audio2exp_model.test(data)['exp_coeff_pred']
        pose_pred = audio2pose_model.test(data)['pose_pred']
        data['driving_coeff'] = torch.cat([exp_coeff_pred, pose_pred], dim=-1)
        os.makedirs(save_dir, exist_ok=True)
        facerender_model.test(data, save_dir)

        #if args.enhance:
        #    predicted_frames = enhancer_list(predicted_frames)

        
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--config",
                        default='./src/config/test.yaml', help="test config file")
    parser.add_argument(
        "--audio2exp_ckpt", default='./exp/audio2exp/latest.pth', help="audio2exp checkpoint file")
    parser.add_argument(
        "--audio2pose_ckpt", default='./exp/audio2pose/latest.pth', help="audio2pose checkpoint file")
    parser.add_argument(
        "--facerender_ckpt", default='./exp/facerender/latest.pth', help="facerender checkpoint file")
    parser.add_argument(
        "--result_dir", default='./results/test', help="path to output")
    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--ngpus', type=int, default=1, help='num gpus.')
    parser.add_argument('--enhance', action='store_true', help='enhance predictions')
    args = parser.parse_args()
    set_random_seed(args.seed)

    if args.ngpus > 1:
        mp.spawn(main, args=(args.ngpus, args), nprocs=args.ngpus)
    else:
        main(0, 1, args)
