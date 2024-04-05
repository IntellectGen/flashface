from typing import Any, Callable, Dict, List, Optional, Union
import os
import cv2
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from diffusers import UniPCMultistepScheduler, DDIMScheduler, UNet2DConditionModel, AutoencoderKL

from referencenet import ReferenceAttentionControl


def make_image_grid(images: List[Image.Image], rows: int, cols: int, resize: int = None) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

class StableDiffusionReferencePipeline(StableDiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
        reference_unet = None
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        self.register_modules(
            reference_unet=reference_unet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.ref_image_processor = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5)),
        ])


    def prepare_image(
        self,
        image: Union[torch.Tensor, Image.Image, List[Union[torch.Tensor, Image.Image]]],
        width: int,
        height: int,
        batch_size: int,
        num_images_per_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
        do_classifier_free_guidance: bool = False,
        guess_mode: bool = False,
    ) -> torch.Tensor:

        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                image = torch.stack([self.ref_image_processor(im) for im in image], dim=0)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_ref_latents(
        self,
        refimage: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[int, List[int]],
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        r"""
        Prepares reference latents for generating images.

        Args:
            refimage (torch.Tensor): The reference image.
            batch_size (int): The desired batch size.
            dtype (torch.dtype): The data type of the tensors.
            device (torch.device): The device to perform computations on.
            generator (int or list): The generator index or a list of generator indices.
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance.

        Returns:
            torch.Tensor: The prepared reference latents.
        """
        refimage = refimage.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            ref_image_latents = [
                self.vae.encode(refimage[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(generator=generator)
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if ref_image_latents.shape[0] < batch_size:
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            ref_image_latents = ref_image_latents.repeat(batch_size // ref_image_latents.shape[0], 1, 1, 1)

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        ref_image: Union[torch.FloatTensor, Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        reference_scale = 1.0

    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=1,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4.2 Preprocess reference image
        ref_image = self.prepare_image(
            image=ref_image,
            width=256,
            height=256,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
        )
    
        # 5.1 Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5.2 Prepare reference latent variables
        ref_image_latents = self.prepare_ref_latents(
            ref_image,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="midup",
            reference_scale=reference_scale
        )
        reference_control_reader = ReferenceAttentionControl(
            self.unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="midup",
            reference_scale=reference_scale
        )
        

        enable_reference = True
        if enable_reference:
            print(f"enable referencenet !")

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # 1. Forward reference image
                if enable_reference and i == 0:
                    n = len(ref_image)
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=prompt_embeds.repeat_interleave(n, dim=0),
                        return_dict=False,
                    )

                    # 2. Update reference unet feature into denosing net
                    reference_control_reader.update(reference_control_writer)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.to(latent_model_input.dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
            reference_control_reader.clear()
            reference_control_writer.clear()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


if __name__ == "__main__":

    device = "cuda"
    dtype = torch.float32

    checkpoint_path = ""
    pipe = ""

    metadata = {}
    metadata["prompt"] = "a woman with a flower in her hair, white dress, looking at viewer, realistic, blue background, hair flower, simple background, upper body"

    print(metadata)


    ref_faces_path = [
        "1.jpg", "2.jpg"
    ]

    ref_faces = [Image.open(i) for i in ref_faces_path]
    make_image_grid(ref_faces, 1, len(ref_faces), resize=256).save("face_pixel_values.jpg")
    empty_face = Image.new("RGB", (256, 256), color=(127, 127, 127))

    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet").to(device)
    reference_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="ref").to(device)

    pipe = StableDiffusionReferencePipeline.from_pretrained(
        pipe,
        unet=unet,
        reference_unet = reference_unet,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, steps_offset=19)
    pipe.set_progress_bar_config(ncols=50)

    num_images_per_prompt = 1

    seed = 12345

    target_width, target_height = 512, 512

    num_refs = [1, 2, 3, 4]
    results = []

    for n in num_refs:
        image = pipe(
            metadata["prompt"],
            width = target_width, height = target_height,
            ref_image = ref_faces[:n],
            num_inference_steps = 50,
            num_images_per_prompt = num_images_per_prompt,
            generator = torch.Generator(device=device).manual_seed(seed),
            guidance_scale = 3.0,
            reference_scale = 0.85,
            negative_prompt = "watermark, text, font, bad eye, cross eye, low quality, worst quality, bad anatomy, bad composition, lowres, poor, low effort, blurry, monochrome, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hands, poorly rendered hands, mutated body parts, blurry image, disfigured, oversaturated, deformed body features",
        ).images[0]
        results.append(image)

    make_image_grid(results, 1, len(num_refs)).save("result.png")

