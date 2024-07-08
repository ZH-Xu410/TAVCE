import cv2
import os
import torch
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import face_alignment

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = "1"  # 设置MKL-DNN CPU加速库的线程数。
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dir):
    # Returns list: [path1, path2, ...]
    image_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_files.append(path)
    return image_files


def save_result(image_pth, landmarks):
    out_pth = image_pth.replace('images', 'landmarks')
    os.makedirs(os.path.dirname(out_pth), exist_ok=True)
    landmark_file = os.path.splitext(out_pth)[0] + '.txt'
    np.savetxt(landmark_file, landmarks)

def dirs_exist(image_pths):
    lnd_pths = [p.replace('images', 'landmarks') for p in image_pths]
    out_paths = set(os.path.dirname(lnd_pth) for lnd_pth in lnd_pths)
    return all([os.path.exists(out_path) for out_path in out_paths])

def detect_landmarks(img_paths, predictor, device):
    prev_points = None
    for i in tqdm(range(len(img_paths)), desc='detect landmarks'):
        if os.path.exists(os.path.splitext(img_paths[i].replace('images', 'landmarks'))[0] + '.txt'):
            continue

        img = io.imread(img_paths[i])
        preds = predictor.get_landmarks_from_image(img)
        if preds is not None:
            #if len(preds)>2:
            #    print('More than one faces were found in %s' % img_paths[i])

            points = preds[0]
            prev_points = points
            
        else:
            print('No face detected, using previous landmarks')
            points = prev_points
        try:
            save_result(img_paths[i], points)
        except:
            pass

def print_args(parser, args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------------------'
    print(message)

def main():
    print('---------- Eye-landmarks detection --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--data_dir', type=str, default='~/datasets/VoxCeleb2/', help='Path to celebrity folder.')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Print Arguments
    print_args(parser, args)

    predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, face_detector='sfd')

    # Get the path of each image.
    image_dir = os.path.join(args.data_dir, 'images')
    #image_paths = sorted(get_image_paths(images_dir))
    image_paths = sorted(glob(image_dir+'/**/*.png', recursive=True))
    #with open('img_list.txt') as f:
    #    image_paths = f.read().splitlines()
    
    imgae_paths = [image_paths[i] for i in range(args.rank, len(img_paths), args.world_size)]
    detect_landmarks(image_paths, predictor, device)
    print('DONE!')

if __name__=='__main__':
    main()
