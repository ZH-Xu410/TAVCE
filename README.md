
# TAVCE - Official Implementation
## Exploiting Temporal Audio-Visual Correlation Embedding for Audio-Driven One-Shot Talking Head Animation



## 1. Installation


Create a conda environment and install the requirements.
  ```bash
conda create -n sadtalker python=3.8

conda activate sadtalker

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

conda install ffmpeg

pip install -r requirements.txt
  ```  


## 2. Download Pretrained Models

- First, download the pretrained models of SadTalker at [here](https://github.com/OpenTalker/SadTalker/releases/tag/v0.0.1).
- Download pretrained TAVCE models at Github Release page.


## 3. Prepare Dataset

Download the dataset (e.g. HDTF). Put video frames under 'HDTF/images' folder.

```bash
# run preprocessing
python preprocess/pipeline.py
```

## 4. Test
```bash
python inference.py --config src/config/test.yaml
```

## 5. Train TAVCE
```bash
python correlation/train.py --data_root ~/datasets/VoxCeleb2 --work_dir exp/correlation
```
Then use the pretrained correlation model to supervise the facerender.


## Acknowledgements

Codes are borrowed heavily from [SadTalker](https://github.com/OpenTalker/SadTalker).  We thank them for their wonderful work.