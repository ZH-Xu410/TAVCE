from src.audio2exp_models import load_audio2exp_model
from src.audio2pose_models import load_audio2pose_model
from src.facerender import load_animate_model, load_face_vid2vid

def load_model(cfg, device, is_train=True, **kwargs):
    if cfg.MODEL.TYPE == 'Audio2Exp':
        return load_audio2exp_model(cfg, device, is_train, **kwargs)
    elif cfg.MODEL.TYPE == 'Audio2Pose':
        return load_audio2pose_model(cfg, device, is_train, **kwargs)
    elif cfg.MODEL.TYPE == 'Animate':
        return load_animate_model(cfg, device, is_train, **kwargs)
    elif cfg.MODEL.TYPE == 'FaceVid2Vid':
        return load_face_vid2vid(cfg, device, is_train, **kwargs)
