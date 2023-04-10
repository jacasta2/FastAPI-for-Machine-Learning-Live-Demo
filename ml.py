# https://huggingface.co/blog/stable_diffusion
# https://huggingface.co/CompVis/stable-diffusion-v1-4

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

from typing import Union

token_path = Path("token_jac.txt") # Make sure to adjust this with the file where you have the token
token = token_path.read_text().strip()
# Get your token at https://huggingface.co/settings/tokens

# Replaced 'torch.float16' for 'torch.float32' to make it work.
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision = "fp16",
    torch_dtype = torch.float32,
    use_auth_token = token,
)

#pipe.to("cuda") # It seems I don't have enough GPU capacity. 

prompt = "a photograph of an astronaut riding a horse"

#image1 = pipe(prompt)["sample"][0] # Kills the kernel.
#image1 = pipe(prompt, height = 384, width = 384).images[0]
#image1

# Replaced 'seed: int | None = None' for 'seed: Union[int, None] = None'.
    # Added 'from typing import Union' to be able to work with the replacement above.
    # However, this Python version (3.10.10) makes this tweak no longer necessary.
# Given my GPU issues, I replaced 'Generator("cuda")' for 'Generator()' and added dimensions (height and
    # width) less than 512.
# I removed the arbitrary arguments stuff (i.e., the '*,') after 'prompt: str,'.
def obtain_image(
    prompt: str,
    seed: Union[int, None] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image:
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    print(f"Using device: {pipe.device}")
    image: Image = pipe(
        prompt,
        height = 384,
        width = 384,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        generator = generator,
    ).images[0]
    return image

#image2 = obtain_image(prompt, num_inference_steps = 25, seed = 1024)
#image2
