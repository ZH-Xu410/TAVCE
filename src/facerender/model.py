import torch
import os
import cv2
import warnings
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.resnet import Bottleneck
from src.facerender.modules.discriminator import MultiScaleDiscriminator
from src.facerender.modules.make_animation import keypoint_transformation
from src.facerender.modules.generator import OcclusionAwareSPADEGenerator
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.hopenet import Hopenet
from src.facerender.modules.vgg19 import Vgg19
from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.util import (
    AntiAliasInterpolation2d,
    make_coordinate_grid_2d,
)
from correlation.models import IResNet18, AudioEncoder
from correlation.utils import fit_ROI_in_frame_batch, crop_ROI_batch
from src.lip_reading import load_model, get_preprocessing_pipelines
from src.face3d.extract_kp_videos_safe import KeypointExtractor

warnings.filterwarnings("ignore")


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


class ImagePyramide(nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace(".", "-")] = AntiAliasInterpolation2d(
                num_channels, scale
            )
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict["prediction_" + str(scale).replace("-", ".")] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """

    def __init__(self, bs, **kwargs):
        noise = torch.normal(
            mean=0, std=kwargs["sigma_affine"] * torch.ones([bs, 2, 3])
        )
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ("sigma_tps" in kwargs) and ("points_tps" in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d(
                (kwargs["points_tps"], kwargs["points_tps"]), type=noise.type()
            )
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(
                mean=0,
                std=kwargs["sigma_tps"]
                * torch.ones([bs, 1, kwargs["points_tps"] ** 2]),
            )
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(
            self.bs, frame.shape[2], frame.shape[3], 2
        )
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = (
            torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1))
            + theta[:, :, :, 2:]
        )
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(
                coordinates.shape[0], -1, 1, 2
            ) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances**2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = torch.autograd.grad(
            new_coordinates[..., 0].sum(), coordinates, create_graph=True
        )
        grad_y = torch.autograd.grad(
            new_coordinates[..., 1].sum(), coordinates, create_graph=True
        )
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 99

    return degree


class AnimateModel(nn.Module):
    def __init__(self, config, device, is_train=True, is_cmtc=False):
        super().__init__()
        self.config = config
        self.generator = OcclusionAwareSPADEGenerator(
            **config["model_params"]["generator_params"],
            **config["model_params"]["common_params"],
        )
        self.kp_detector = KPDetector(
            **config["model_params"]["kp_detector_params"],
            **config["model_params"]["common_params"],
        )
        self.he_estimator = HEEstimator(
            **config["model_params"]["he_estimator_params"],
            **config["model_params"]["common_params"],
        )
        self.mapping = MappingNet(**config["model_params"]["mapping_params"])
        self.gen_scales = config["train_params"]["scales"]
        self.disc_scales = config["model_params"]["discriminator_params"]["scales"]
        self.gen_pyramid = ImagePyramide(self.gen_scales, self.generator.image_channel)
        self.disc_pyramid = ImagePyramide(
            self.disc_scales, self.generator.image_channel
        )

        self.device = device
        self.is_cmtc = is_cmtc

        if self.is_cmtc:
            self.image_encoder = IResNet18(num_features=128).cuda()
            self.audio_encoder = AudioEncoder()
            ckpt = torch.load(config["train_params"]["cmtc_image_ckpt"], "cpu")
            self.image_encoder.load_state_dict(
                {
                    k.replace("module.", ""): v
                    for k, v in ckpt["image_encoder"].items()
                }
            )
            self.audio_encoder.load_state_dict(
                {
                    k.replace("module.", ""): v
                    for k, v in ckpt["audio_encoder"].items()
                }
            )
            self.image_encoder.eval().to(device)
            self.audio_encoder.eval().to(device)
            self.no_grad(self.image_encoder)
            self.no_grad(self.audio_encoder)

    def no_grad(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def load_from_facevid2vid(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"], strict=False)
        self.kp_detector.load_state_dict(checkpoint["kp_detector"])
        self.he_estimator.load_state_dict(checkpoint["he_estimator"])
        self.mapping.load_state_dict(checkpoint["mapping"])
        if hasattr(self, "discriminator"):
            self.discriminator.load_state_dict(checkpoint["discriminator"])


    def forward(self, x):
        # {'value': value, 'jacobian': jacobian}
        kp_canonical = self.kp_detector(x["source_image"])

        # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
        he_source = self.mapping(x["source_coeff"][:, :70])
        he_driving = self.mapping(x["driving_coeff"][:, :70])

        # {'value': value, 'jacobian': jacobian}
        kp_source = keypoint_transformation(kp_canonical, he_source)
        kp_driving = keypoint_transformation(kp_canonical, he_driving)

        if self.is_cmtc:
            audios = x["indiv_mels"]
            bs, _, ca, ha, wa = audios.shape
            audio_features = self.audio_encoder(audios.view(bs * 2, ca, ha, wa))
            audio_features = audio_features.view(bs, 2, -1)
            audio_correlation = cov_matrix(
                audio_features[:, 0], audio_features[:, 1]
            )

            x["cor"] = audio_correlation.detach()

        generated = self.generator(
            x["source_image"],
            kp_source=kp_source,
            kp_driving=kp_driving,
            cor=x.get("cor", None),
        )
        generated.update(
            {
                "kp_canonical": kp_canonical,
                "he_source": he_source,
                "he_driving": he_driving,
                "kp_source": kp_source,
                "kp_driving": kp_driving,
            }
        )

        if self.is_cmtc:
            try:
                with torch.no_grad():
                    lm = np.stack(
                        [
                            self.kp_extractor.extract_keypoint(
                                (255 * img)
                                .detach()
                                .cpu()
                                .numpy()
                                .transpose((1, 2, 0))
                                .astype(np.uint8)
                            )
                            for img in generated["prediction"]
                        ],
                        axis=0,
                    )
                    pred_centers = torch.from_numpy(
                        np.median(lm[:, 48:], axis=1).astype(np.int32)
                    )
            except:
                return generated
            last_frames = crop_ROI_batch(
                x["last_frame"], fit_ROI_in_frame_batch(x["mouth_centers"][:, 0])
            )
            predictions = crop_ROI_batch(
                generated["prediction"], fit_ROI_in_frame_batch(pred_centers)
            )
            gt_mouth = crop_ROI_batch(
                x["driving_image"], fit_ROI_in_frame_batch(x["mouth_centers"][:, 1])
            )
            generated["pred_mouth"] = predictions
            generated["gt_mouth"] = gt_mouth

            images = (torch.stack([last_frames, predictions], dim=1) - 0.5) / 0.5

            bs, _, ci, hi, wi = images.shape
            image_features = self.image_encoder(
                F.interpolate(
                    images.view(bs * 2, ci, hi, wi), (112, 112), mode="bilinear"
                )
            )
            image_features = image_features.view(bs, 2, -1)

            image_correlation = cov_matrix(
                image_features[:, 0].detach(),
                image_features[:, 1],
            )

            generated["audio_correlation"] = audio_correlation.view(bs, -1)
            generated["image_correlation"] = image_correlation.view(bs, -1)

        return generated

    def load(self, ckpt):
        ckpt = torch.load(ckpt, self.device)
        self.mapping.load_state_dict(ckpt["mapping"])
        self.generator.load_state_dict(ckpt["generator"], strict=False)
        if hasattr(self, "discriminator"):
            self.discriminator.load_state_dict(ckpt["discriminator"])
        self.kp_detector.load_state_dict(ckpt["kp_detector"])
        self.he_estimator.load_state_dict(ckpt["he_estimator"])

    def test(self, x, save_dir, frame_len=27, batch_size=1):
        T = x["source_image"].shape[1]

        source_image = x["source_image"][:, 0].repeat(batch_size, 1, 1, 1)
        kp_canonical = self.kp_detector(source_image)

        he_source = self.mapping(
            x["ref"][:1, :1, :70].repeat(batch_size, frame_len, 1).permute(0, 2, 1)
        )
        kp_source = keypoint_transformation(kp_canonical, he_source)

        for i in tqdm(range(0, T, batch_size), "generator"):
            driving_coeffs = []
            correlations = []
            for j in range(i, min(i + batch_size, T)):
                frame_idx = [
                    min(max(0, k), T - 1)
                    for k in range(i - frame_len // 2, i + frame_len // 2 + 1)
                ]
                driving_coeffs.append(
                    x["driving_coeff"][:, frame_idx, :70].permute(0, 2, 1)
                )
                if x.get("cor") is not None:
                    correlations.append(x["cor"][:, j])
            driving_coeffs = torch.cat(driving_coeffs, dim=0)
            correlations = torch.cat(correlations, dim=0) if len(correlations) else None

            he_driving = self.mapping(driving_coeffs)

            # if x.get("yaw_seq") is not None:
            #     he_driving["yaw_in"] = x["yaw_seq"][:, frame_idx]
            # if x.get("pitch_seq") is not None:
            #     he_driving["pitch_in"] = x["pitch_seq"][:, frame_idx]
            # if x.get("roll_seq") is not None:
            #     he_driving["roll_in"] = x["roll_seq"][:, frame_idx]

            kp_driving = keypoint_transformation(kp_canonical, he_driving)

            generated = self.generator(
                source_image,
                kp_source=kp_source,
                kp_driving=kp_driving,
                cor=correlations,
            )
            out_img = generated["prediction"] * 255
            out_img = (
                out_img.permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                .astype(np.uint8)[:, :, :, ::-1]
            )
            for j, img in enumerate(out_img):
                cv2.imwrite(os.path.join(save_dir, f"{i+j:04d}.png"), img)
