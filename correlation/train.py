import argparse
import os
import cv2
import time
import torch
import warnings
import lpips
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from correlation.models import IResNet18, AudioEncoder, MVal, Generator
import src.utils.audio as audio_util
from src.generate_batch import parse_audio_length, crop_pad_audio
from src.utils.croper import Preprocesser
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from correlation.utils import fit_ROI_in_frame, crop_ROI

log_file = None


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav = audio_util.load_wav(audio_path, 16000)
        return wav


class AlignedDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.data_root = args.data_root
        self.load_size = args.load_size
        self.fps = args.fps
        self.tau = args.tau
        self.id_sampling = args.id_sampling
        if self.id_sampling:
            videos = list({os.path.basename(video).split('#')[0] 
                                for video in os.listdir(self.data_root)})
        else:
            videos = os.listdir(self.data_root)
        self.videos = videos * args.num_repeats
        
        LOG(f'{len(self.videos)} samples.')

    def __len__(self):
        return len(self.videos)

    def _load_images(self, images):
        batch = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.load_size, self.load_size))
            #image = image.astype(np.float32) / 255
            #image = (image - 0.5) / 0.5
            batch.append(image)
        batch = np.stack(batch, axis=0)  # n,h,w,c
        # batch = np.transpose(batch, (0, 3, 1, 2))  # n,c,h,w
        batch = torch.from_numpy(batch.astype(np.float32))
        return batch
    
    def _try_once(self, idx):
        if self.id_sampling:
            name = self.videos[idx]
            video_path = random.choice(glob(os.path.join(self.data_root, name + '*.mp4')))
        else:
            name = self.videos[idx]
            video_path = os.path.join(self.data_root, name)
        audio_path = video_path.replace('videos', 'audios').replace('.mp4', '.wav')
        if not os.path.exists(audio_path):
            return None, video_path
        
        try:
            wav = load_audio(audio_path)
        except:
            return None, audio_path
        
        wav_length, nframes = parse_audio_length(len(wav), 16000, self.fps)
        cap = cv2.VideoCapture(video_path)
        nframes = min(nframes, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if nframes < self.tau + 3:
            return None, video_path        
    
        i = random.randint(1, nframes-2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret1, img1 = cap.read()
        ret2, img2 = cap.read()

        j = random.randint(1, nframes-1)
        while abs(j - i) < self.tau:
            j = random.randint(1, nframes-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ret3, img3 = cap.read()
        if not (ret1 and ret2 and ret3):
            return None, video_path
        
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio_util.melspectrogram(wav).T
        spec = orig_mel.copy()         # nframes 80
        indiv_mels = []
        for k in (i, i+1, j):
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
        audios = torch.from_numpy(indiv_mels.astype(np.float32))

        return {'images': images, 'audios': audios}, video_path 

    def __getitem__(self, idx):
        k = idx
        while True:
            data, path = self._try_once(k)
            if data is not None:
                return data
            else:
                print(f'error for loading {path}, skip')
                k = random.randint(0, len(self.videos)-1)


def LOG(*args, **kwargs):
    if get_rank() != 0:
        return
    _LOG(time.strftime('%Y-%m-%d#%H:%M:%S', time.localtime(time.time())), end=' ')
    _LOG(*args, **kwargs)
    log_file.flush()


def _LOG(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)


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


def tensor2image(tensor):
    image = (tensor * 0.5 + 0.5) * 255
    image = image.permute(1, 2, 0).cpu().numpy()
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return image


def main(rank, world_size, args):
    global log_file
    if world_size > 1:
        ddp_setup(rank, world_size)
    if rank == 0:
        vis_dir = os.path.join(args.work_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        log_file = open(os.path.join(args.work_dir, 'loss_log.txt'), 'w')

    image_encoder = IResNet18(num_features=args.feat_dim).cuda()
    audio_encoder = AudioEncoder(dim=args.feat_dim).cuda()
    generator1 = Generator(128, 3, 3).cuda()
    generator2 = Generator(128, 3, 3).cuda()
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.bs,
                            num_workers=args.num_workers, persistent_workers=True,
                            sampler=DistributedSampler(
                                dataset) if world_size > 1 else None,
                            shuffle=world_size == 1)
    optimizer = torch.optim.Adam(
        list(image_encoder.parameters())+list(audio_encoder.parameters()) +
        list(generator1.parameters())+list(generator2.parameters()),
        lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, 0.1)
    if world_size > 1:
        image_encoder = DDP(image_encoder, device_ids=[rank])
        audio_encoder = DDP(audio_encoder, device_ids=[rank])
        generator1 = DDP(generator1, device_ids=[rank])
        generator2 = DDP(generator2, device_ids=[rank])
    loss_fn_vgg =  lpips.LPIPS(net='vgg').cuda(rank) #torch.nn.L1Loss()
    pos_sim = MVal()
    neg_sim = MVal()
    preprocesser = Preprocesser(f'cuda:{rank}')

    for epoch in range(args.epochs):
        for step, data in enumerate(dataloader):
            images, audios = data['images'], data['audios']
            bs, ni, hi, wi, ci = images.shape
            images_pil = [Image.fromarray(img.numpy().astype(np.uint8)) for img in images.view(bs*ni, hi, wi, ci)]
            try:
                lms = preprocesser.predictor.extract_keypoint(images_pil, info=False)
            except:
                continue

            cropped_imgs = []
            for img, lm in zip(images_pil, lms):
                if lm is None or lm.mean() == -1:
                    continue
                mouth_center = np.median(lm[48:], axis=0).astype(np.int32)
                img = crop_ROI(np.array(img), fit_ROI_in_frame(mouth_center))
                img = cv2.resize(img, (args.crop_size, args.crop_size))
                cropped_imgs.append(img)
            
            if len(cropped_imgs) != bs*ni:
                continue
            hi, wi = args.crop_size, args.crop_size
            images = np.stack(cropped_imgs, axis=0).reshape(bs, ni, hi, wi, ci)
            images = torch.from_numpy(images.astype(np.float32)).cuda()
            images = images.permute(0, 1, 4, 2, 3).contiguous()/255  # bs ni ci hi wi
            images = (images - 0.5) / 0.5

            audios = audios.cuda()
            bs, na, ca, ha, wa = audios.shape

            audio_embeddings = audio_encoder(audios.view(bs*na, ca, ha, wa))
            audio_embeddings = audio_embeddings.view(bs, na, -1)
            image_embeddings = image_encoder(images.view(bs*ni, ci, hi, wi))
            image_embeddings = image_embeddings.view(bs, ni, -1)

            audio_correlation = cov_matrix(
                audio_embeddings[:, 0], audio_embeddings[:, 1])
            image_correlation1 = cov_matrix(
                image_embeddings[:, 0], image_embeddings[:, 1])
            image_correlation2 = cov_matrix(
                image_embeddings[:, 0], image_embeddings[:, 2])

            recon_next = generator1(audio_correlation, images[:, 0])
            recon_prev = generator2(audio_correlation, images[:, 1])

            loss_pos = F.cosine_embedding_loss(audio_correlation.view(bs, -1), image_correlation1.view(bs, -1), torch.ones(bs, dtype=torch.int64, device=audio_embeddings.device))
            loss_neg = F.cosine_embedding_loss(audio_correlation.view(bs, -1), image_correlation2.view(bs, -1), -torch.ones(bs, dtype=torch.int64, device=audio_embeddings.device))
            loss_recon = loss_fn_vgg(recon_next, images[:, 1]).mean() + loss_fn_vgg(recon_prev, images[:, 0]).mean()
            loss = loss_pos + args.alpha * loss_neg + args.lambd * loss_recon
            #loss = loss_recon_img + loss_recon_aud
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sim_pos = F.cosine_similarity(
                audio_correlation, image_correlation1, dim=-1).mean()
            sim_neg = F.cosine_similarity(
                audio_correlation, image_correlation2, dim=-1).mean()

            pos_sim.update(sim_pos.item())
            neg_sim.update(sim_neg.item())

            if rank == 0 and (step + 1) % args.log_interval == 0:
                LOG(f'epoch {epoch+1}, step [{step+1}/{len(dataloader)}], loss {loss.item():.4f}, '
                    f'l_pos: {loss_pos.item():.4f}, l_neg: {loss_neg.item():.4f}, '
                    f'l_rec: {loss_recon.item():.4f}, '
                    f'lr {optimizer.param_groups[0]["lr"]:.4g}, '
                    f'pos sim: {pos_sim.val:.4f}, neg sim: {neg_sim.val:.4f}')
                vis_recon_next = tensor2image(recon_next[0].detach())
                vis_recon_prev = tensor2image(recon_prev[0].detach())
                vis_real_next = tensor2image(images[0, 1])
                vis_real_prev = tensor2image(images[0, 0])
                cv2.imwrite(os.path.join(
                    vis_dir, f'recon_next_epoch{epoch}.png'), vis_recon_next)
                cv2.imwrite(os.path.join(
                    vis_dir, f'recon_prev_epoch{epoch}.png'), vis_recon_prev)
                cv2.imwrite(os.path.join(
                    vis_dir, f'real_next_epoch{epoch}.png'), vis_real_next)
                cv2.imwrite(os.path.join(
                    vis_dir, f'real_prev_epoch{epoch}.png'), vis_real_prev)

        lr_scheduler.step()
        if rank == 0:
            ckpt = {
                'image_encoder': image_encoder.state_dict(),
                'audio_encoder': audio_encoder.state_dict(),
                'generator1': generator1.state_dict(),
                'generator2': generator2.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(ckpt, os.path.join(args.work_dir, f'epoch_{epoch}.pth'))
    LOG('Done.')
    if world_size > 1:
        destroy_process_group()
    if rank == 0:
        log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='~/datasets/VoxCeleb2/videos', help='data root')
    parser.add_argument('--id_sampling', action='store_true', help='sample training ids')
    parser.add_argument('--num_repeats', type=int, default=1, help='number of repeats for dataset')
    parser.add_argument('--load_size', type=int, default=256, help='image size')
    parser.add_argument('--crop_size', type=int, default=112, help='image size')
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='dim for feature vector')
    parser.add_argument('--margin', type=int, default=0)
    parser.add_argument('--fps', type=int, default=25, help='audio fps')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step', type=int, default=8)
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--work_dir', type=str,
                        default='exp/correlation')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--lambd', type=float, default=1.)
    parser.add_argument('--tau', type=int, default=3)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.ngpus > 1:
        mp.spawn(main, args=(args.ngpus, args), nprocs=args.ngpus)
    else:
        main(0, 1, args)
