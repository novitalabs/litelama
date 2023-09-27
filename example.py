from litelama import LiteLama
import requests
from PIL import Image
from io import BytesIO
import time


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"


lama = LiteLama()
lama.load(use_safetensors=True)
lama.to("cuda:0")
init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

stime = time.time()
for i in range(100):
    lama.predict(init_image, mask_image)

print(f"Time: {(time.time() - stime) / 100} per image")
