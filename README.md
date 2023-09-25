# Lite Lama - A lightweight LAMA inference wrapper

```python
from litelama import LiteLama
import requests
from PIL import Image
from io import BytesIO


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"


lama = LiteLama()
lama.to("cuda:0")
init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

lama.predict(init_image, mask_image).save("result.png")
```