import os
import json
import torch
import pickle
import random
import numpy as np
import scipy.io as scio
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
import src.utils.audio as audio_util
from src.generate_batch import parse_audio_length, crop_pad_audio, generate_blink_seq_randomly


class VoxAudioDataset(Dataset):
    def __init__(self, root_dir, frame_len, hop_len=1, name='audio2exp', audio_subdir='audios/train', coeff_subdir='coeffs/train', sr=16000, fps=25, 
                 mel_filter_banks=80, syncnet_mel_step_size=16, return_class=True, eye_blink=True, num_audios=1890):
        self.root_dir = root_dir
        self.audio_subdir = audio_subdir
        self.coeff_subdir = coeff_subdir
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.sr = sr
        self.fps = fps
        self.mel_filter_banks = mel_filter_banks
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.num_audios = num_audios
        self.return_class = return_class
        self.eye_blink = eye_blink
        self.cache_path = f'.cache/vox_audio_dataset-{name}.pkl'
        self.audios = []
        self.coeffs = []
        self.full_coeffs = []
        self.refs = []
        self.blinks = []
        self.classes = []

        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            check_code = json.dumps({'frame_len': frame_len, 'hop_len': self.hop_len, 'coeff_subdir': coeff_subdir})
            if cache_data['check_code'] == check_code:
                print('load dataset from cache')
                self.audios = cache_data['audios']
                self.coeffs = cache_data['coeffs']
                self.full_coeffs = cache_data['full_coeffs']
                self.refs = cache_data['refs']
                self.blinks = cache_data.get('blinks', [])
                self.classes = cache_data.get('classes', [])
            else:
                self.load_data()
        else:
            self.load_data()
        print(f'{len(self.audios)} samples loaded')
    
    def load_data(self):
        print('loading data')
        audio_list = list(glob(os.path.join(self.root_dir, self.audio_subdir, '**/*.wav'), recursive=True))
        all_ids = sorted(os.listdir(os.path.join(self.root_dir, self.audio_subdir)))
        if len(audio_list) > self.num_audios:
            audio_list = random.sample(audio_list, self.num_audios)

        for a in tqdm(audio_list):
            p = a.split(self.audio_subdir)[1]
            if p.startswith('/'):
                p = p[1:]
            id_, track, name = p.split('/')
            name = name.split('.')[0]
            coeff_dir = os.path.join(self.root_dir, self.coeff_subdir, id_, track, name)
            if not os.path.exists(coeff_dir):
                continue

            wav = audio_util.load_wav(a, self.sr) 
            wav_length, num_frames = parse_audio_length(len(wav), self.sr, self.fps)
            wav = crop_pad_audio(wav, wav_length)
            orig_mel = audio_util.melspectrogram(wav).T
            spec = orig_mel.copy()         # nframes 80
            indiv_mels = []

            if self.eye_blink:
                blinks = generate_blink_seq_randomly(num_frames).astype(np.float32)

            for i in range(num_frames):
                start_frame_num = i-2
                start_idx = int(self.mel_filter_banks * (start_frame_num / float(self.fps)))
                end_idx = start_idx + self.syncnet_mel_step_size
                seq = list(range(start_idx, end_idx))
                seq = [min(max(item, 0), orig_mel.shape[0]-1) for item in seq]
                m = spec[seq, :]
                indiv_mels.append(m.T)
            # indiv_mels = np.asarray(indiv_mels)         # T 80 16
            coeff_files = sorted([f for f in os.listdir(coeff_dir) if f.endswith('.mat')])
            coeffs = [scio.loadmat(os.path.join(coeff_dir, c))['coeff_3dmm'] for c in coeff_files]
            coeffs = [coeffs[int(i)] for i in np.linspace(0, len(coeffs)-1, num=num_frames)]
            full_coeffs = [scio.loadmat(os.path.join(coeff_dir, c))['full_3dmm'] for c in coeff_files]
            full_coeffs = [full_coeffs[int(i)] for i in np.linspace(0, len(full_coeffs)-1, num=num_frames)]

            for i in range(0, num_frames-self.frame_len, self.hop_len):
                self.audios.append(np.array(indiv_mels[i:i+self.frame_len]).astype(np.float32))
                self.coeffs.append(np.concatenate(coeffs[i:i+self.frame_len], axis=0).astype(np.float32))
                self.full_coeffs.append(np.concatenate(full_coeffs[i:i+self.frame_len], axis=0).astype(np.float32))
                self.refs.append(np.repeat(coeffs[0], self.frame_len, axis=0).astype(np.float32))
                self.classes.append(all_ids.index(id_))
                if self.eye_blink:
                    self.blinks.append(blinks[i:i+self.frame_len])
        
        check_code = json.dumps({'frame_len': self.frame_len, 'hop_len': self.hop_len, 'coeff_subdir': self.coeff_subdir})
        cache_data = {'check_code': check_code,
                      'audios': self.audios,
                      'coeffs': self.coeffs,
                      'full_coeffs': self.full_coeffs,
                      'refs': self.refs,
                      'blinks': self.blinks,
                      'classes': self.classes}
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = torch.from_numpy(self.audios[idx]).unsqueeze(1) # T 1 80 16 
        coeff = torch.from_numpy(self.coeffs[idx]) # T d
        full_coeff = torch.from_numpy(self.full_coeffs[idx])
        ref = torch.from_numpy(self.refs[idx])
        data = {'indiv_mels': audio, 'ref': ref, 'coeff_gt': coeff, 'full_coeff': full_coeff}
        if self.return_class:
            data['class'] = torch.LongTensor([self.classes[idx]])
        if self.eye_blink:
            blink = torch.from_numpy(self.blinks[idx]) # T 1
            data['ratio_gt'] = blink

        return data


if __name__ == '__main__':
    random.seed(0)
    dataset = VoxAudioDataset('~/datasets/VoxCeleb1', 5, 5)
    data = dataset[0]
    for k, v in data.items():
        print(k, v.shape)
    
