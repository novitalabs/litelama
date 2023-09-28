import requests
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import torch.nn as nn
from pathlib import Path
import yaml
import os
from .ffc import FFCResNetGenerator
from .pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, \
    NLayerDiscriminator, MultidilatedNLayerDiscriminator
import safetensors

import logging

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")


def download_file(url, path):
    logging.info(f"Downloading {url} to {path}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        try:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            f.close()
            if os.path.exists(path) and os.path.isfile(path):
                os.remove(path)
            raise e


def load_model(config_path: str = None, checkpoint_path: str = None, use_safetensors: bool = True):
    if config_path is None:
        config_path = os.path.join(PROJECT_PATH, "config.yaml")

    if checkpoint_path is None:
        home = Path.home()
        tmp_ = home.joinpath(".cache/litelama")
        tmp_.mkdir(parents=True, exist_ok=True)
        if use_safetensors:
            checkpoint_path = str(tmp_.joinpath("big-lama.safetensors"))
        else:
            checkpoint_path = str(tmp_.joinpath("big-lama.ckpt"))

        if not os.path.exists(checkpoint_path):
            if use_safetensors:
                download_file("https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors", checkpoint_path)
            else:
                download_file("https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.ckpt", checkpoint_path)

    with open(config_path, "r") as f:
        config = OmegaConf.create(yaml.safe_load(f))

    model = DefaultInpaintingTrainingModule(config)
    if use_safetensors:
        state_dict = {}
        with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict=state_dict, strict=False)
    return model


def make_generator(config, kind, **kwargs):
    if kind == 'pix2pixhd_multidilated':
        return MultiDilatedGlobalGenerator(**kwargs)

    if kind == 'pix2pixhd_global':
        return GlobalGenerator(**kwargs)

    if kind == 'ffc_resnet':
        return FFCResNetGenerator(**kwargs)

    raise ValueError(f'Unknown generator kind {kind}')


def make_multiscale_noise(base_tensor, scales=6, scale_mode='bilinear'):
    batch_size, _, height, width = base_tensor.shape
    cur_height, cur_width = height, width
    result = []
    align_corners = False if scale_mode in ('bilinear', 'bicubic') else None
    for _ in range(scales):
        cur_sample = torch.randn(batch_size, 1, cur_height, cur_width, device=base_tensor.device)
        cur_sample_scaled = F.interpolate(cur_sample, size=(height, width), mode=scale_mode, align_corners=align_corners)
        result.append(cur_sample_scaled)
        cur_height //= 2
        cur_width //= 2
    return torch.cat(result, dim=1)


class DefaultInpaintingTrainingModule(nn.Module):
    def __init__(self, config, concat_mask=True):
        super().__init__()

        config.predict_only = True
        self.config = config
        self.generator = make_generator(config, **self.config.generator)
        self.add_noise_kwargs = None
        self.concat_mask = concat_mask

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']

        masked_img = img * (1 - mask)

        if self.add_noise_kwargs is not None:
            noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

        return batch
