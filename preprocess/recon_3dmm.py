import argparse
import os
import torch
import glob
import tqdm
import warnings
import safetensors
import safetensors.torch
import numpy as np
from PIL import Image
from src.face3d.util.preprocess import align_img
from src.face3d.util.load_mats import load_lm3d
from src.face3d.models import networks
from scipy.io import loadmat, savemat
from src.utils.safetensor_helper import load_x_from_safetensor

warnings.filterwarnings("ignore")

img_exts = [".jpg", ".png"]


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


def recon_3dmm(img, lm3d_std, net_recon, device, no_save=False):
    save_dir = os.path.dirname(img).replace("images", "coeffs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(img).split(".")[0] + ".mat")
    if os.path.exists(save_path):
        return
    try:
        frame = Image.open(img)
    except:
        os.remove(img)
        return
    W, H = frame.size
    lm = np.loadtxt(
        os.path.splitext(img.replace("images", "landmarks"))[0] + ".txt"
    ).reshape((-1, 2))
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

    result = {"coeff_3dmm": pred_coeff, "full_3dmm": full_coeff.cpu().numpy()}
    if not no_save:
        savemat(save_path, result)
    
    return result


def main():
    # paths = {
    #     'use_safetensor': True,
    #     'checkpoint': 'checkpoints/SadTalker_V0.0.2_'+str(args.size)+'.safetensors',
    #     'dir_of_BFM_fitting': 'src/config'
    # }
    # model = CropAndExtract(paths, device)
    imgs = []
    for e in img_exts:
        imgs += list(
            glob.glob(os.path.join(args.data_root, f"**/*{e}"), recursive=True)
        )
    imgs = sorted(imgs)
    if len(args.keys):
        imgs = list(filter(lambda x: any([k in x for k in args.keys]), imgs))

    inds = list(range(args.rank, len(imgs), args.world_size))
    print(f"rank: {args.rank}, num imgs: {len(inds)}")

    for i in tqdm.tqdm(inds):
        img = imgs[i]
        recon_3dmm(img, lm3d_std=lm3d_std, net_recon=net_recon, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/2022_stu/zihua/datasets/VoxCeleb2-1000ID/images",
        help="image root",
    )
    parser.add_argument("--size", type=int, default=256, help="image size")
    parser.add_argument("--world_size", type=int, default=1, help="world_size")
    parser.add_argument("--rank", type=int, default=0, help="rank")
    parser.add_argument(
        "--keys", type=str, nargs="+", default=[], help="keys to filter images"
    )
    args = parser.parse_args()

    sadtalker_path = {
        "use_safetensor": True,
        "checkpoint": "checkpoints/SadTalker_V0.0.2_" + str(args.size) + ".safetensors",
        "dir_of_BFM_fitting": "src/config",
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_recon = networks.define_net_recon(
        net_recon="resnet50", use_last_fc=False, init_path=""
    ).to(device)
    checkpoint = safetensors.torch.load_file(sadtalker_path["checkpoint"])
    net_recon.load_state_dict(load_x_from_safetensor(checkpoint, "face_3drecon"))
    net_recon.eval()
    lm3d_std = load_lm3d(sadtalker_path["dir_of_BFM_fitting"])

    main()
