import os
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import torch
import argparse
from facenet_pytorch import MTCNN, extract_face
from tqdm import tqdm
from glob import glob
from shutil import rmtree

VID_EXTENSIONS = ['.mp4']

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)

def tensor2npimage(image_tensor, imtype=np.uint8):
    # Tesnor in range [0,255]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2npimage(image_tensor[i], imtype))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, transpose = True):
    if transpose:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_images(images, folder, name, args):
    for i in range(len(images)):
        n_frame = "{:06d}".format(i)
        save_dir = os.path.join(args.data_root, folder, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(images[i], os.path.join(save_dir, n_frame + '.png'), transpose = folder =='images')


def get_video_paths(dir):
    # Returns list of paths to video files
    video_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_video_file(fname):
                path = os.path.join(root, fname)
                video_files.append(path)
    return video_files


def smooth_boxes(boxes, previous_box, args):
    # Check if there are None boxes.
    if boxes[0] is None:
        boxes[0] = previous_box
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = next((item for item in boxes[i+1:] if item is not None), boxes[i-1])
    boxes = [box[0] for box in boxes if box is not None]   # if more than one faces detected, keep the one with the heighest probability
    if len(boxes) == 0:
        return []

    # Smoothen boxes
    old_boxes = np.array(boxes)
    window_length = min(args.window_length, old_boxes.shape[0])
    if window_length % 2 == 0:
        window_length -= 1

    # import pdb;pdb.set_trace()

    #smooth_boxes = np.concatenate([ndimage.median_filter(old_boxes[:,i].astype(np.float32), size=window_length, mode='reflect').reshape((-1,1)) for i in range(4)], 1)
    min_x0 = old_boxes[:, 0].min()
    min_y0 = old_boxes[:, 1].min()
    max_x1 = old_boxes[:, 2].max()
    max_y1 = old_boxes[:, 3].max()

    smooth_boxes = np.array([min_x0, min_y0, max_x1, max_y1])
    offset_w = smooth_boxes[2] - smooth_boxes[0]
    offset_h = smooth_boxes[3] - smooth_boxes[1]
    offset_dif = (offset_h - offset_w) / 2
    # width
    smooth_boxes[0] = smooth_boxes[2] - offset_w - offset_dif
    smooth_boxes[2] = smooth_boxes[2] + offset_dif
    # height - center a bit lower
    smooth_boxes[3] = smooth_boxes[3] + args.height_recentre * offset_h
    smooth_boxes[1] = smooth_boxes[3] - offset_h
    smooth_boxes = smooth_boxes[np.newaxis, :].repeat(len(old_boxes), axis=0)
    # Make boxes square.
    # for i in range(len(smooth_boxes)):
    #     offset_w = smooth_boxes[i][2] - smooth_boxes[i][0]
    #     offset_h = smooth_boxes[i][3] - smooth_boxes[i][1]
    #     offset_dif = (offset_h - offset_w) / 2
    #     # width
    #     smooth_boxes[i][0] = smooth_boxes[i][2] - offset_w - offset_dif
    #     smooth_boxes[i][2] = smooth_boxes[i][2] + offset_dif
    #     # height - center a bit lower
    #     smooth_boxes[i][3] = smooth_boxes[i][3] + args.height_recentre * offset_h
    #     smooth_boxes[i][1] = smooth_boxes[i][3] - offset_h

    return smooth_boxes

def smooth_boxes2(boxes, previous_box, args):
    # Check if there are None boxes.
    if boxes[0] is None:
        boxes[0] = previous_box
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = next((item for item in boxes[i+1:] if item is not None), boxes[i-1])
    boxes = [box[0] for box in boxes if box is not None]   # if more than one faces detected, keep the one with the heighest probability
    if len(boxes) == 0:
        return []

    # Smoothen boxes
    old_boxes = np.array(boxes)
    window_length = min(args.window_length, old_boxes.shape[0])
    if window_length % 2 == 0:
        window_length -= 1

    # import pdb;pdb.set_trace()

    smooth_boxes = np.concatenate([ndimage.median_filter(old_boxes[:,i].astype(np.float32), size=window_length, mode='reflect').reshape((-1,1)) for i in range(4)], 1)

    # Make boxes square.
    for i in range(len(smooth_boxes)):
        offset_w = smooth_boxes[i][2] - smooth_boxes[i][0]
        offset_h = smooth_boxes[i][3] - smooth_boxes[i][1]
        offset_dif = (offset_h - offset_w) / 2
        # width
        smooth_boxes[i][0] = smooth_boxes[i][2] - offset_w - offset_dif
        smooth_boxes[i][2] = smooth_boxes[i][2] + offset_dif
        # height - center a bit lower
        smooth_boxes[i][3] = smooth_boxes[i][3] + args.height_recentre * offset_h
        smooth_boxes[i][1] = smooth_boxes[i][3] - offset_h

    return smooth_boxes

def get_faces(detector, images, previous_box, args):
    ret_faces = []
    ret_boxes = []

    all_boxes = []
    all_imgs = []

    # Get bounding boxes
    for lb in np.arange(0, len(images), args.mtcnn_batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+args.mtcnn_batch_size]]
        boxes, _ = detector.detect(imgs_pil)
        all_boxes.extend(boxes)
        all_imgs.extend(imgs_pil)
    
    if len(all_boxes) == 0:
        return [], [], None
    
    # Temporal smoothing
    boxes = smooth_boxes(all_boxes, previous_box, args)

    if len(boxes) == 0:
        return [], [], None

    # Crop face regions.
    for img, box in zip(all_imgs, boxes):
        face = extract_face(img, box, args.cropped_image_size, args.margin)
        ret_faces.append(face)
        # Find real bbox   (taken from https://github.com/timesler/facenet-pytorch/blob/54c869c51e0e3e12f7f92f551cdd2ecd164e2443/models/utils/detect_face.py#L358)
        margin = [
            args.margin * (box[2] - box[0]) / (args.cropped_image_size - args.margin),
            args.margin * (box[3] - box[1]) / (args.cropped_image_size - args.margin),
        ]
        raw_image_size = img.size
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]
        ret_boxes.append(box)

    return ret_faces, ret_boxes, boxes[-1]


def detect_imgs_and_save_faces(detector, img_dir, args):
    imgs = sorted(list(glob(os.path.join(img_dir, '*.png'))))
    n_frames = len(imgs)
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    name = os.path.splitext(os.path.basename(img_dir))[0]

    images = []
    previous_box = None

    # print('Reading %s, extracting faces, and saving images' % mp4_path)
    for i in tqdm(range(n_frames), desc='detect face'):
        image = cv2.imread(imgs[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(images) < args.filter_length:
            images.append(image)
        # else, detect faces in sequence and create new list
        else:
            face_images, boxes, previous_box = get_faces(detector, images, previous_box, args)
            if len(face_images) > 0:
                save_images(tensor2npimage(face_images), 'images', name, args)

                if args.save_full_frames:
                    save_images(images, 'full_frames', name, args)

            if args.save_videos_info:
                videos_file = os.path.splitext(mp4_path)[0] + '.txt'
                if not os.path.exists(videos_file):
                    vfile = open(videos_file, "a")
                    vfile.write('{} {} fps {} frames\n'.format(mp4_path, fps, n_frames))
                    vfile.close()
                for box in boxes:
                    vfile = open(videos_file, "a")
                    np.savetxt(vfile, np.expand_dims(box,0))
                    vfile.close()

            images = [image]
    # last sequence
    face_images, boxes, _ = get_faces(detector, images, previous_box, args)
    save_images(tensor2npimage(face_images), 'images', name, args)

    if args.save_full_frames:
        save_images(images, 'full_frames', name, args)

    if args.save_videos_info:
        videos_file = os.path.splitext(mp4_path)[0] + '.txt'
        if not os.path.exists(videos_file):
            vfile = open(videos_file, "a")
            vfile.write('{} {} fps {} frames\n'.format(mp4_path, fps, n_frames))
            vfile.close()
        for box in boxes:
            vfile = open(videos_file, "a")
            np.savetxt(vfile, np.expand_dims(box,0))
            vfile.close()


def detect_and_save_faces(detector, mp4_path, args):

    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    name = os.path.splitext(os.path.basename(mp4_path))[0]

    images = []
    previous_box = None

    # print('Reading %s, extracting faces, and saving images' % mp4_path)
    for i in tqdm(range(n_frames), desc='detect face'):
        _, image = reader.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(images) < args.filter_length:
            images.append(image)
        # else, detect faces in sequence and create new list
        else:
            face_images, boxes, previous_box = get_faces(detector, images, previous_box, args)
            if len(face_images) > 0:
                save_images(tensor2npimage(face_images), 'images', name, args)

                if args.save_full_frames:
                    save_images(images, 'full_frames', name, args)

            if args.save_videos_info:
                videos_file = os.path.splitext(mp4_path)[0] + '.txt'
                if not os.path.exists(videos_file):
                    vfile = open(videos_file, "a")
                    vfile.write('{} {} fps {} frames\n'.format(mp4_path, fps, n_frames))
                    vfile.close()
                for box in boxes:
                    vfile = open(videos_file, "a")
                    np.savetxt(vfile, np.expand_dims(box,0))
                    vfile.close()

            images = [image]
    # last sequence
    face_images, boxes, _ = get_faces(detector, images, previous_box, args)
    save_images(tensor2npimage(face_images), 'images', name, args)

    if args.save_full_frames:
        save_images(images, 'full_frames', name, args)

    if args.save_videos_info:
        videos_file = os.path.splitext(mp4_path)[0] + '.txt'
        if not os.path.exists(videos_file):
            vfile = open(videos_file, "a")
            vfile.write('{} {} fps {} frames\n'.format(mp4_path, fps, n_frames))
            vfile.close()
        for box in boxes:
            vfile = open(videos_file, "a")
            np.savetxt(vfile, np.expand_dims(box,0))
            vfile.close()

    reader.release()


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
    print('-------------- Face detection -------------- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--data_root', type=str, default='~/datasets/HDTF/videos', help='Path to celebrity folder.')
    parser.add_argument('--save_videos_info', action='store_true', help='Whether to save videos meta-data (fps, #frames, bounding boxes) in .txt file')
    parser.add_argument('--save_full_frames', action='store_true', help='Whether to save full video frames (for reproducing the original clip)')
    parser.add_argument('--mtcnn_batch_size', default=64, type=int, help='The number of frames for face detection.')
    parser.add_argument('--select_largest', action='store_true', help='In case of multiple detected faces, keep the largest (if specified), or the one with the highest probability')
    parser.add_argument('--cropped_image_size', default=256, type=int, help='The size of frames after cropping the face.')
    parser.add_argument('--margin', default=30, type=int, help='.')
    parser.add_argument('--filter_length', default=500, type=int, help='Number of consecutive bounding boxes to be filtered')
    parser.add_argument("--max_frames", default=-1, type=int, help="Number of maximum frames")
    parser.add_argument('--window_length', default=49, type=int, help='savgol filter window length.')
    parser.add_argument('--height_recentre', default=0.0, type=float, help='The amount of re-centring bounding boxes lower on the face.')
    parser.add_argument('--seq_length', default=50, type=int, help='The number of frames for each training sub-sequence.')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    args = parser.parse_args()
    print_args(parser, args)

    # check if face detection has already been done
    images_dir = os.path.join(args.data_root, 'images')


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
    
    # subfolder containing videos
    videos_path = os.path.join(args.data_root, 'videos')

    # Store video paths in list.
    mp4_paths = sorted(get_video_paths(videos_path))
    mp4_paths = [mp4_paths[i] for i in range(args.rank, len(mp4_paths), args.world_size)]
    n_mp4s = len(mp4_paths)
    print('Number of videos to process: %d \n' % n_mp4s)

    # Initialize the MTCNN face  detector.
    detector = MTCNN(image_size=args.cropped_image_size, select_largest = args.select_largest, margin=args.margin, post_process=False, device=device)

    # Run detection
    n_completed = 0
    for path in mp4_paths:
        if not os.path.exists(os.path.join(args.data_root, 'images',  os.path.splitext(os.path.basename(path))[0])):
            detect_and_save_faces(detector, path, args)
        n_completed += 1
        print('(%d/%d) %s [SUCCESS]' % (n_completed, n_mp4s, path))

    print('DONE!')

if __name__ == "__main__":
    main()
