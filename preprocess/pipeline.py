import os
import cv2
import time
import random
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = "1"  # 设置MKL-DNN CPU加速库的线程数。

import torch

torch.set_num_threads(1)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import argparse
import safetensors
import safetensors.torch
import face_alignment
from PIL import Image
from glob import glob
from tqdm import tqdm
from scipy import ndimage
from shutil import rmtree
from facenet_pytorch import MTCNN, extract_face
from preprocess.detect_face import detect_imgs_and_save_faces, detect_and_save_faces
from preprocess.detect_landmarks import detect_landmarks
from preprocess.recon_3dmm import recon_3dmm
from src.face3d.models import networks
from src.face3d.util.load_mats import load_lm3d
from src.utils.safetensor_helper import load_x_from_safetensor


def print_args(parser, args):
    message = ""
    message += "----------------- Arguments ---------------\n"
    for k, v in sorted(vars(args).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = "\t[default: %s]" % str(default)
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "-------------------------------------------"
    print(message)


def main(mode="train"):
    mp4_paths = glob(os.path.join(args.data_root, f"videos/*"))
    random.seed(2024)
    random.shuffle(mp4_paths)
    mp4_paths = [
        mp4_paths[i] for i in range(args.rank, len(mp4_paths), args.world_size)
    ]
    n_mp4s = len(mp4_paths)
    print("Number of videos to process: %d \n" % n_mp4s)

    # Run detection
    n_completed = 0
    for path in mp4_paths:
        save_name = os.path.basename(path).rsplit(".", 1)[0]
        if not os.path.exists(os.path.join(args.data_root, "images", save_name)):
            try:
                if os.path.isdir(path):
                    detect_imgs_and_save_faces(detector, path, args)
                else:
                    detect_and_save_faces(detector, path, args)
            except:
                continue
        
        if mode == "train":
            if not os.path.exists(os.path.join(args.data_root, "landmarks", save_name)):
                try:
                    img_paths = glob(
                        os.path.join(args.data_root, "images", save_name, "*")
                    )
                    detect_landmarks(img_paths, predictor, device)
                except:
                    continue
            if not os.path.exists(os.path.join(args.data_root, "coeffs", save_name)):
                img_paths = glob(os.path.join(args.data_root, "images", save_name, "*"))
                for img in tqdm(img_paths, desc="recon 3dmm"):
                    try:
                        recon_3dmm(img, lm3d_std, net_recon, device)
                    except:
                        pass
        n_completed += 1
        t = time.strftime("%Y-%m-%d#%H:%M:%S", time.localtime(time.time()))
        print("%s (%d/%d) %s [SUCCESS]" % (t, n_completed, n_mp4s, path))

    print("DONE!")


if __name__ == "__main__":
    print("-------------- Preprocess -------------- \n")
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="Negative value to use CPU, or greater or equal than zero for GPU id.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="~/datasets/HDTF/",
        help="Path to celebrity folder.",
    )
    parser.add_argument(
        "--save_videos_info",
        action="store_true",
        help="Whether to save videos meta-data (fps, #frames, bounding boxes) in .txt file",
    )
    parser.add_argument(
        "--save_full_frames",
        action="store_true",
        help="Whether to save full video frames (for reproducing the original clip)",
    )
    parser.add_argument(
        "--mtcnn_batch_size",
        default=64,
        type=int,
        help="The number of frames for face detection.",
    )
    parser.add_argument(
        "--select_largest",
        action="store_true",
        help="In case of multiple detected faces, keep the largest (if specified), or the one with the highest probability",
    )
    parser.add_argument(
        "--cropped_image_size",
        default=256,
        type=int,
        help="The size of frames after cropping the face.",
    )
    parser.add_argument("--margin", default=70, type=int, help=".")
    parser.add_argument(
        "--filter_length",
        default=200, # 500
        type=int,
        help="Number of consecutive bounding boxes to be filtered",
    )
    parser.add_argument(
        "--max_frames",
        default=200,
        type=int,
        help="Number of maximum frames",
    )
    parser.add_argument(
        "--window_length", default=49, type=int, help="savgol filter window length."
    )
    parser.add_argument(
        "--height_recentre",
        default=0.0,
        type=float,
        help="The amount of re-centring bounding boxes lower on the face.",
    )
    parser.add_argument(
        "--seq_length",
        default=50,
        type=int,
        help="The number of frames for each training sub-sequence.",
    )
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    print_args(parser, args)

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = "cpu"
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = "cuda:0"
        else:
            device = "cuda:" + str(gpu_id)
    else:
        print("GPU device not available. Exit")
        exit(0)

    detector = MTCNN(
        image_size=args.cropped_image_size,
        select_largest=args.select_largest,
        margin=args.margin,
        post_process=False,
        device=device,
    )

    predictor = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=device, face_detector="sfd"
    )

    sadtalker_path = {
        "use_safetensor": True,
        "checkpoint": "checkpoints/SadTalker_V0.0.2_256.safetensors",
        "dir_of_BFM_fitting": "src/config",
    }
    net_recon = networks.define_net_recon(
        net_recon="resnet50", use_last_fc=False, init_path=""
    ).to(device)
    checkpoint = safetensors.torch.load_file(sadtalker_path["checkpoint"])
    net_recon.load_state_dict(load_x_from_safetensor(checkpoint, "face_3drecon"))
    net_recon.eval()
    lm3d_std = load_lm3d(sadtalker_path["dir_of_BFM_fitting"])

    # main('train')
    main('test')
