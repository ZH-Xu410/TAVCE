from torch.utils.data.dataloader import DataLoader
from src.datasets.vox_audio_dataset import VoxAudioDataset
from src.datasets.vox_frame_dataset import VoxFrameDataset
from src.datasets.test_dataset import TestDataset
from src.datasets.vox_cmtc_dataset import VoxCMTCDataset
from src.datasets.vox_video_dataset import VoxVideoDataset
from torch.utils.data.distributed import DistributedSampler


def get_dataset(cfg):
    if cfg.TYPE == 'VoxAudio':
        if cfg.NAME == 'audio2pose':
            return_class = True
            eye_blink = False
        else:  # 'audio2exp'
            return_class = False
            eye_blink = True
        return VoxAudioDataset(cfg.ROOT_DIR, cfg.FRAME_LEN, cfg.HOP_LEN, cfg.NAME,
                               return_class=return_class, eye_blink=eye_blink)
    elif cfg.TYPE == 'VoxFrame':
        return VoxFrameDataset(cfg.ROOT_DIR, cfg.IMG_SIZE, cfg.FRAME_LEN)
    
    elif cfg.TYPE == 'Test':
        return TestDataset(cfg.ROOT_DIR, cfg.KEYS, cfg.IMG_SIZE)
    
    elif cfg.TYPE == 'VoxCMTC':
        return VoxCMTCDataset(cfg.ROOT_DIR, cfg.FRAME_LEN, cfg.ID_SAMPLING, img_size=cfg.IMG_SIZE, num_repeats=cfg.NUM_REPEATS, selected_ids=cfg.SELECTED_IDS, img_dirs=cfg.IMG_DIRS)
    
    elif cfg.TYPE == 'VoxVideo':
        return VoxVideoDataset(cfg.ROOT_DIR, cfg.ID_SAMPLING, cfg.IMG_SIZE, num_repeats=cfg.NUM_REPEATS)


def get_dataloader(cfg, ddp=False):
    dataset = get_dataset(cfg.DATASET)
    if ddp:
        dataloader = DataLoader(dataset, batch_size=cfg.TRAIN_BATCH_SIZE, sampler=DistributedSampler(dataset),
                                num_workers=cfg.NUM_WORKERS, drop_last=True, pin_memory=False)
    else:
        dataloader = DataLoader(dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True,
                                num_workers=cfg.NUM_WORKERS, drop_last=True, pin_memory=False)
    return dataloader


def get_test_dataloader(cfg, ddp=False):
    dataset = get_dataset(cfg.DATASET)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, drop_last=False, sampler=DistributedSampler(dataset) if ddp else None)
    return dataloader