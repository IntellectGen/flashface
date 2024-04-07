from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image

import torch
from diffusers import DDIMScheduler
from pipeline import make_image_grid, StableDiffusionReferencePipeline

if __name__ == "__main__":

    device = "cuda"
    dtype = torch.float16
    pipe = "IntellectGen/FlashFace"

    seed = 12345
    width, height = 512, 512
    prompt = "a woman with a flower in her hair, white dress, looking at viewer, realistic, blue background, hair flower, simple background, upper body"
    prompt = "asian face, " + prompt

    ref_faces_path = [
        "asset/yangmi1.face.jpg", 
        "asset/yangmi2.face.jpg", 
        "asset/yangmi3.face.jpg", 
        "asset/yangmi4.face.jpg", 
    ]

    pipe = StableDiffusionReferencePipeline.from_pretrained(pipe, torch_dtype=dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, steps_offset=19)
    pipe.set_progress_bar_config(ncols=50)

    results: List[Image.Image] = []
    ref_faces = [Image.open(i) for i in ref_faces_path]
    make_image_grid(ref_faces, 1, len(ref_faces), resize=256).save("faces.jpg")

    for n in range(1, 1+len(ref_faces)):
        image = pipe(
            prompt,
            width = width, height = height,
            ref_image = ref_faces[:n],
            num_inference_steps = 50,
            num_images_per_prompt = 1,
            generator = torch.Generator(device=device).manual_seed(seed),
            guidance_scale = 3.0,
            reference_scale = 0.85,
            negative_prompt = "watermark, text, font, bad eye, cross eye, low quality, worst quality, bad anatomy, bad composition, lowres, poor, low effort, blurry, monochrome, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hands, poorly rendered hands, mutated body parts, blurry image, disfigured, oversaturated, deformed body features",
            enable_reference = True
        ).images[0]
        results.append(image)

    make_image_grid(results, 1, len(ref_faces_path)).save("viz.png")
    results[-1].save("result.png")
