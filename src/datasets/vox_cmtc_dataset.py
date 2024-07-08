import os
import io
import cv2
import json
import torch
import pickle
import random
import warnings
import requests
import numpy as np
import scipy.io as scio
from glob import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import quote, unquote
from skimage import img_as_float32
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Compose, ColorJitter
import src.utils.audio as audio_util
from src.generate_batch import parse_audio_length, crop_pad_audio, generate_blink_seq_randomly


def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert os.path.exists(audio_path)
        wav = audio_util.load_wav(audio_path, 16000)
    return wav


def load_txt_data(txt_path):
    return np.loadtxt(txt_path, np.float32)


def load_img(img_path):
    return cv2.imread(img_path)


def load_mat(mat_path):
    return scio.loadmat(mat_path)


class VoxCMTCDataset(Dataset):
    def __init__(self, root_dir, frame_len=27, id_sampling=False, selected_ids=None, img_dirs=['images/train'],
                 img_size=256, sr=16000, fps=25, mel_filter_banks=80, syncnet_mel_step_size=16, num_repeats=1):
        self.root_dir = root_dir
        self.frame_len = frame_len
        self.img_size = img_size
        self.sr = sr
        self.fps = fps
        self.mel_filter_banks = mel_filter_banks
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.id_sampling = id_sampling
        self.img_dirs = img_dirs

        self.all_videos = []
        for img_dir in self.img_dirs:
            self.all_videos += os.listdir(os.path.join(root_dir, img_dir))

        if id_sampling:
            if selected_ids:
                with open(selected_ids) as f:
                    self.videos = f.read().splitlines()
            else:
                self.videos = list({os.path.basename(video).split('#')[0] for video in os.listdir(os.path.join(root_dir, img_dirs[0]))})
        else:
            self.videos = self.all_videos
            
        self.videos = self.videos * num_repeats

        print(f'{len(self.videos)} samples')

    def _load_images(self, images):
        batch = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = img_as_float32(image)
            batch.append(image)
        batch = np.stack(batch, axis=0)  # n,h,w,c
        batch = np.transpose(batch, (0, 3, 1, 2))  # n,c,h,w
        batch = torch.from_numpy(batch.astype(np.float32))
        return batch
    
    def __len__(self):
        return len(self.videos)
    
    def _get_img_dir(self):
        if len(self.img_dirs) == 1:
            return self.img_dirs[0]
        else:
            return random.sample(self.img_dirs, 1)[0]

    def _try_once(self, idx):
        img_dir = self._get_img_dir()
        if self.id_sampling:
            name = self.videos[idx]
            videos = list(filter(lambda x: x.startswith(name), self.all_videos))
            if len(videos) == 0:
                return None, name
            name = random.choice(videos)
        else:
            name = self.videos[idx]
        
        video_path = os.path.join(self.root_dir, img_dir, name)
        audio_path = video_path.replace(img_dir.split('/')[0], 'audios') + '.wav'

        try:
            wav = load_audio(audio_path)
        except:
            return None, audio_path
        
        frames = sorted(os.listdir(video_path))
        frames = [os.path.join(video_path, f) for f in frames if f.endswith('.png')]
        wav_length, nframes = parse_audio_length(len(wav), 16000, self.fps)
        nframes = min(nframes, len(frames))
        if nframes < 3:
            return None, video_path        
    
        i = random.randint(0, nframes-1)
        img1 = load_img(frames[i])

        j = random.randint(0, nframes-2)
        img2 = load_img(frames[j])
        img3 = load_img(frames[j+1])

        if img1 is None or img2 is None or img3 is None:
            return None, video_path
        
        images = self._load_images([img1, img2, img3])
        
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio_util.melspectrogram(wav).T
        spec = orig_mel.copy()         # nframes 80
        indiv_mels = []
        for k in (j, j+1):
            start_frame_num = k-2
            start_idx = int(
                80 * (start_frame_num / float(self.fps)))
            end_idx = start_idx + 16
            seq = list(range(start_idx, end_idx))
            seq = [min(max(item, 0), orig_mel.shape[0]-1) for item in seq]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)[:, np.newaxis]  # T 1 80 16
        indiv_mels = torch.from_numpy(indiv_mels.astype(np.float32))

        frame_id = j+1
        coeff_list = []
        coeff_dir = video_path.replace('images', 'coeffs')
        coeffs = os.listdir(coeff_dir)
        coeffs = {int(x.split('.')[0]): os.path.join(coeff_dir, x) for x in coeffs if x.endswith('.mat')}
        for k in range(frame_id - self.frame_len//2, frame_id + self.frame_len//2 + 1):
            k = min(max(k, 0), nframes-1)
            if k not in coeffs:
                break
            try:
                coeff = load_mat(coeffs[k])['coeff_3dmm']
            except:
                break
            coeff_list.append(coeff)
        
        if len(coeff_list) != self.frame_len or i not in coeffs:
            return None, video_path

        coeff = np.concatenate(coeff_list, axis=0).transpose((1, 0)).astype(np.float32)
        src_coeff = load_mat(coeffs[i])['coeff_3dmm'].reshape(-1, 1)
        src_coeff = np.repeat(src_coeff, self.frame_len, axis=1)

        try:
            lm0 = load_txt_data(frames[j].replace('images', 'landmarks').replace('.png', '.txt')).reshape((-1, 2))
            lm1 = load_txt_data(frames[j+1].replace('images', 'landmarks').replace('.png', '.txt')).reshape((-1, 2))
            mouth_center = np.stack([np.median(lm0[48:], axis=0), np.median(lm1[48:], axis=0)], axis=0).astype(np.int32)
        except:
            return None, video_path

        return {'source_image': images[0], 'driving_image': images[2], 'last_frame': images[1], 
                'indiv_mels': indiv_mels, 'source_coeff': src_coeff, 'driving_coeff': coeff,
                'mouth_centers': mouth_center}, video_path 

    def __getitem__(self, idx):
        k = idx
        while True:
            data, path = self._try_once(k)
            if data is not None:
                return data
            else:
                print(f'error for loading {path}, skip')
                k = random.randint(0, len(self.videos)-1)


if __name__ == '__main__':
    random.seed(0)
    dataset = VoxCMTCDataset('~/datasets/VoxCeleb1', 32, 5)
    data = dataset[0]
    for k, v in data.items():
        print(k, v.shape)
