import os
import cv2
import time
import glob
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from itertools import cycle
from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.detection import init_detection_model
from torch.multiprocessing import Pool, Process, set_start_method


class KeypointExtractor():
    def __init__(self, device='cuda'):

        ### gfpgan/weights
        try:
            import webui  # in webui
            root_path = 'extensions/SadTalker/gfpgan/weights' 

        except:
            root_path = 'gfpgan/weights'

        self.detector = init_alignment_model('awing_fan',device=device, model_rootpath=root_path)   
        self.det_net = init_detection_model('retinaface_resnet50', half=False,device=device, model_rootpath=root_path)

    def extract_keypoint(self, images, name=None, info=True, return_box=False):
        if isinstance(images, list):
            keypoints = []
            bboxes = []
            if info:
                i_range = tqdm(images,desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                current_kp, current_box = self.extract_keypoint(image, return_box=True)
                # current_kp = self.detector.get_landmarks(np.array(image))
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                    bboxes.append(bboxes[-1])
                else:
                    keypoints.append(current_kp[None])
                    bboxes.append(current_box[None])

            keypoints = np.concatenate(keypoints, 0)
            bboxes = np.concatenate(bboxes, 0)
            if name is not None:
                np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            if return_box:
                return keypoints, bboxes
            return keypoints
        else:
            while True:
                try:
                    with torch.no_grad():
                        # face detection -> face alignment.
                        img = np.array(images)
                        bboxes = self.det_net.detect_faces(images, 0.5) # 0.97
                        bboxes = np.clip(bboxes[0], a_min=0, a_max=img.shape[0])
                        img = img[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]

                        keypoints = landmark_98_to_68(self.detector.get_landmarks(img)) # [0]

                        #### keypoints to the original location
                        keypoints[:,0] += int(bboxes[0])
                        keypoints[:,1] += int(bboxes[1])

                        break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break    
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)         
                    bboxes = [0, 0, images.shape[:2]]
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            if return_box:
                return keypoints, bboxes
            return keypoints

def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    name = filename.split('/')[-2:]
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    kp_extractor.extract_keypoint(
        images, 
        name=os.path.join(opt.output_dir, name[-2], name[-1])
    )

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)

    opt = parser.parse_args()
    filenames = list()
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS
    
    for ext in extensions:
        os.listdir(f'{opt.input_dir}')
        print(f'{opt.input_dir}/*.{ext}')
        filenames = sorted(glob.glob(f'{opt.input_dir}/*.{ext}'))
    print('Total number of videos:', len(filenames))
    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None
