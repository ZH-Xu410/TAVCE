import os
import torch
import random
import numpy as np
import scipy.io as scio
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import src.utils.audio as audio_util
from src.generate_batch import parse_audio_length, crop_pad_audio, generate_blink_seq_randomly


class TestDataset(Dataset):
    def __init__(self, root_dir, keys, img_size=256, audio_subdir='audios', image_subdir='images', sr=16000, fps=25,
                 mel_filter_banks=80, syncnet_mel_step_size=16, pose_style='constant', eye_blink='random', max_num=-1):
        self.root_dir = root_dir
        self.audio_subdir = audio_subdir
        self.image_subdir = image_subdir
        self.keys = keys
        self.img_size = img_size
        self.sr = sr
        self.fps = fps
        self.mel_filter_banks = mel_filter_banks
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.audio_list = list(
            glob(os.path.join(self.root_dir, audio_subdir, '**/*.wav'), recursive=True))
        if len(self.keys):
            self.audio_list = list(filter(lambda x: any(
                [k in x for k in self.keys]), self.audio_list))
        if max_num > 0 and len(self.audio_list) > max_num:
            self.audio_list = random.sample(self.audio_list, max_num)
        print(f'{len(self.audio_list)} videos for test')

        self.transform = ToTensor()
        self.pose_style = pose_style
        self.eye_blink = eye_blink

    def __len__(self):
        return len(self.audio_list)

    def _load_img(self, img):
        img = Image.open(img)
        img = img.resize((self.img_size, self.img_size))
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        audio = self.audio_list[idx]
        image_dir = audio.replace(self.audio_subdir, self.image_subdir).rsplit('.', 1)[0]
        name = image_dir.split(self.image_subdir+'/')[1]

        wav = audio_util.load_wav(audio, self.sr)
        wav_length, num_frames = parse_audio_length(
            len(wav), self.sr, self.fps)
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio_util.melspectrogram(wav).T
        spec = orig_mel.copy()         # nframes 80
        indiv_mels = []

        frames = sorted([f for f in os.listdir(
            image_dir) if f.endswith('.png')])
        num_frames = min(num_frames, len(frames))

        for i in range(num_frames):
            start_frame_num = i-2
            start_idx = int(self.mel_filter_banks *
                            (start_frame_num / float(self.fps)))
            end_idx = start_idx + self.syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [min(max(item, 0), orig_mel.shape[0]-1) for item in seq]
            m = spec[seq, :]
            indiv_mels.append(m.T)

        indiv_mels = torch.from_numpy(np.asarray(
            indiv_mels)).float().unsqueeze(1)         # T 1 80 16

        imgs = torch.stack([self._load_img(os.path.join(image_dir, i))
                           for i in frames[:num_frames]], dim=0)

        if self.pose_style == 'random':
            pose_style = random.randint(0, 45)
        else:  # constant
            pose_style = 0

        if self.eye_blink == 'random':
            ratio = torch.from_numpy(
                generate_blink_seq_randomly(num_frames)).float()
        else:
            ratio = torch.zeros([num_frames], dtype=torch.int64)

        data = {'indiv_mels': indiv_mels, 'source_image': imgs,
                'class': pose_style, 'ratio_gt': ratio, 'num_frames': num_frames, 'name': name}

        return data


if __name__ == '__main__':
    random.seed(0)
    dataset = TestDataset('~/HDTF', [
                          'Radio34_9', 'Radio57_0', 'TammyBaldwin1_0', 'BrianSchatz1_0', 'CarlyFiorina_0', 'MikeEnzi_0'])
    data = dataset[0]
    for k, v in data.items():
        print(k, v.shape)
