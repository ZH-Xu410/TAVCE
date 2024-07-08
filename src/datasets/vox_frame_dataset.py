import os
import pickle
import random
import torch
import numpy as np
import scipy.io as scio
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, ColorJitter


class VoxFrameDataset(Dataset):
    def __init__(self, root_dir, img_size, frame_len, name='facerender', coeff_subdir='coeffs/train/'):
        self.root_dir = root_dir
        self.coeff_subdir = coeff_subdir
        self.frame_len = frame_len
        self.img_size = img_size
        self.cache_path = f'.cache/vox_frame_dataset-{name}.pkl'
        self.image_paths = []
        self.coeffs = []
        self.infos = []

        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                print('load dataset from cache')
                self.image_paths = cache_data['image_paths']
                self.coeffs = cache_data['coeffs']
                self.infos = cache_data['infos']
        else:
            self.load_data()
        
        self.idx_group_by_id = {}
        for i, info in enumerate(self.infos):
            track_video = info['id'] + '-' + info['track'] + '-' + info['video']
            if track_video in self.idx_group_by_id:
                self.idx_group_by_id[track_video].append(i)
            else:
                self.idx_group_by_id[track_video] = [i]

        print(f'{len(self.image_paths)} samples loaded')

        self.transform = Compose([
            # RandomHorizontalFlip(),
            # ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor()
        ])
    
    def load_data(self):
        print('loading data')
        img_list = list(glob(os.path.join(self.root_dir, self.coeff_subdir, '**/*.png'), recursive=True))
        all_coeffs = {}
        
        for img in tqdm(img_list):
            p = img.split(self.coeff_subdir)[1]
            if p.startswith('/'):
                p = p[1:]
            id_, track, video, name = p.split('/')
            frame_id = int(name.split('.')[0])
            coeff_dir = os.path.join(self.root_dir, self.coeff_subdir, id_, track, video)
            max_id = int(sorted([f for f in os.listdir(coeff_dir) if f.endswith('.mat')])[-1].split('.')[0])
            coeff_list = []
            for i in range(frame_id - self.frame_len//2, frame_id + self.frame_len//2 + 1):
                i = min(max(i, 1), max_id)
                coeff_path = os.path.join(coeff_dir, f'{i:06d}.mat')
                if not os.path.exists(coeff_path):
                    break
                if coeff_path not in all_coeffs:
                    coeff = scio.loadmat(coeff_path)['coeff_3dmm']
                    all_coeffs[coeff_path] = coeff
                else:
                    coeff = all_coeffs[coeff_path]
                coeff_list.append(coeff)
            if len(coeff_list) != self.frame_len:
                continue
            coeff = np.concatenate(coeff_list, axis=0).transpose((1, 0)).astype(np.float32)
            self.image_paths.append(img)
            self.coeffs.append(coeff)
            self.infos.append({'id': id_, 'track': track, 'video': video, 'frame_id': frame_id})

        cache_data = {'image_paths': self.image_paths,
                      'coeffs': self.coeffs,
                      'infos': self.infos}
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def __len__(self):
        return len(self.image_paths)
    
    def _get_data(self, idx):
        img = Image.open(self.image_paths[idx])
        img = img.resize((self.img_size, self.img_size))
        img = self.transform(img)
        coeff = torch.from_numpy(self.coeffs[idx]).squeeze(0) # d, t 
        return img, coeff

    def __getitem__(self, idx):
        dst_img, dst_coeff = self._get_data(idx)
        info = self.infos[idx]
        track_video = info['id'] + '-' + info['track'] + '-' + info['video']
        idx2 = random.sample(self.idx_group_by_id[track_video], 1)[0]
        src_img, src_coeff = self._get_data(idx2)
        src_coeff = src_coeff[:, src_coeff.shape[1]//2].unsqueeze(1).repeat(1, src_coeff.shape[1])

        data = {'source_image': src_img, 'source_coeff': src_coeff,
                'driving_image': dst_img, 'driving_coeff': dst_coeff}

        return data


if __name__ == '__main__':
    random.seed(0)
    dataset = VoxFrameDataset('~/datasets/VoxCeleb1', 256, 27)
    data = dataset[0]
    for k, v in data.items():
        print(k, v.shape)
