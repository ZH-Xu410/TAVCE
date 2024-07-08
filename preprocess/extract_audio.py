import os
import sys
import argparse
from tqdm import tqdm
from glob import glob
# from moviepy.editor import AudioFileClip


class IgnorePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def extract_audio(video):
    with IgnorePrints():
        audio_path = video.replace(".mp4", ".wav").replace("videos", "audios")
        if os.path.exists(audio_path):
            return
        try:
            #audio_file_clip = AudioFileClip(video)
            #audio_file_clip.write_audiofile(audio_path)
            os.system(f'ffmpeg -i {video} -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -f wav {audio_path}')
        except:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='~/datasets/HDTF/', help='Path to celebrity folder.')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()

    videos = sorted(glob(os.path.join(args.data_root, 'videos/*.mp4')))
    videos = [videos[i] for i in range(args.rank, len(videos), args.world_size)]
    os.makedirs(os.path.join(args.data_root, 'audios'), exist_ok=True)

    for v in tqdm(videos):
        extract_audio(v)
