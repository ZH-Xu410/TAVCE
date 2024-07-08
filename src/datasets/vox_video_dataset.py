import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import img_as_float32
from torch.utils.data import Dataset
import src.utils.audio as audio_util
from src.generate_batch import parse_audio_length, crop_pad_audio
import warnings


def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav = audio_util.load_wav(audio_path, 16000)
        return wav


class VoxVideoDataset(Dataset):
    def __init__(self, root_dir, id_sampling=True, img_size=256, sr=16000, fps=25, mel_filter_banks=80, syncnet_mel_step_size=16, num_repeats=75):
        self.root_dir = root_dir
        self.img_size = img_size
        self.sr = sr
        self.fps = fps
        self.mel_filter_banks = mel_filter_banks
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.id_sampling = id_sampling
        self.num_repeats = num_repeats
        if id_sampling:
            self.videos = list({os.path.basename(video).split('#')[0] 
                                for video in os.listdir(root_dir)})
        else:
            self.videos = os.listdir(root_dir)
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

    def _try_once(self, idx):
        if self.id_sampling:
            name = self.videos[idx]
            video_path = random.choice(glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            video_path = os.path.join(self.root_dir, name)
        audio_path = video_path.replace('videos', 'audios').replace('images', 'audios')
        audio_path = audio_path.replace('.mp4', '.wav') if video_path.endswith('.mp4') else audio_path + '.wav'
        if not os.path.exists(audio_path):
            return None, video_path
        
        try:
            wav = load_audio(audio_path)
        except:
            return None, audio_path
        
        wav_length, nframes = parse_audio_length(len(wav), 16000, self.fps)

        if video_path.endswith('.mp4'):
            cap = cv2.VideoCapture(video_path)
            nframes = min(nframes, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            assert os.path.isdir(video_path)
            frames = sorted(glob(os.path.join(video_path, '*.png')))
            nframes = len(frames)
        if nframes < 3:
            return None, video_path
    
        i = random.randint(1, nframes-1)
        j = random.randint(1, nframes-2)
        if video_path.endswith('.mp4'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret1, img1 = cap.read()
        
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            ret2, img2 = cap.read()
            ret3, img3 = cap.read()
            if not (ret1 and ret2 and ret3):
                return None, video_path
        else:
            img1 = cv2.imread(frames[i])
            img2 = cv2.imread(frames[j])
            img3 = cv2.imread(frames[j+1])
        
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
        
        images = self._load_images([img1, img2, img3])
        indiv_mels = torch.from_numpy(indiv_mels.astype(np.float32))

        return {'source_image': images[0], 'driving_image': images[2], 'last_frame': images[1], 'indiv_mels': indiv_mels}, video_path 

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
    dataset = VoxVideoDataset('~/datasets/VoxCeleb2/videos')
    data = dataset[0]
    for k, v in data.items():
        print(k, v.shape)
