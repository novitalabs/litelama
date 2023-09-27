from litelama.model import load_model
from typing import Optional
from PIL import ImageOps, Image
import torch
import numpy as np
import cv2


class LiteLama:
    def __init__(self, checkpoint_path=None, config_path=None):
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path
        self._model = None

    def load(self, location="cuda", use_safetensors=True):
        if self._model is not None:
            raise RuntimeError("Model is already loaded")

        self._model = load_model(config_path=self._config_path, checkpoint_path=self._checkpoint_path, use_safetensors=use_safetensors)
        self._model.eval()
        self._model.to(location)

    def to(self, location="cuda"):
        if self._model is None:
            return self.load(location)

        self._model.to(location)

    def unload(self):
        self._model = None
        del self._model
        self._model = None

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        image, alpha_channel, exif_infos = load_image(image, return_exif=True)
        mask, _ = load_image(mask, gray=True)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        image = pad_img_to_modulo(
            image, mod=8, square=False
        )
        mask = pad_img_to_modulo(
            mask, mod=8, square=False
        )
        image = norm_img(image)
        mask = norm_img(mask)
        mask = (mask > 0) * 1

        with torch.no_grad():
            image = torch.from_numpy(image).unsqueeze(0).to(self._model.device)
            mask = torch.from_numpy(mask).unsqueeze(0).to(self._model.device)

            res = self._model({
                "image": image,
                "mask": mask
            })

        inpainted_image = res["inpainted"]
        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)

        res_np_img = cv2.cvtColor(cur_res.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if alpha_channel is not None:
            if alpha_channel.shape[:2] != cur_res.shape[:2]:
                alpha_channel = cv2.resize(
                    alpha_channel, dsize=(cur_res.shape[1], res_np_img.shape[0])
                )
            res_np_img = np.concatenate(
                (cur_res, alpha_channel[:, :, np.newaxis]), axis=-1
            )
        return Image.fromarray(res_np_img)


def load_image(image, gray: bool = False, return_exif: bool = False):
    alpha_channel = None
    if return_exif:
        info = image.info or {}
        exif_infos = {"exif": image.getexif(), "parameters": info.get("parameters")}

    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_exif:
        return np_img, alpha_channel, exif_infos
    return np_img, alpha_channel


def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
):

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img
