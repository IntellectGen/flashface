from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import os
import random
import argparse
import itertools
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from arbdata import collate_fn, AspectDataset, AspectSampler
from referencenet import ReferenceAttentionControl

torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.1, help="empty prompt probability")
    parser.add_argument("--proportion_empty_face", type=float, default=0.0, help="empty face probability")
    parser.add_argument("--clip_skip", type=int, default=1, help="text clip encoder skip layer")
    parser.add_argument("--resume_path", type=str, help="resume from checkpoint")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--metafiles",
        type=str,
        required=True,
        nargs="*",
        help="Training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

class ReferenceNet(nn.Module):
    def __init__(
        self,
        denoising_unet: UNet2DConditionModel,
        referencenet: UNet2DConditionModel,
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
        self.referencenet = referencenet
        self.reference_control_writer  = ReferenceAttentionControl(
            referencenet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="midup",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="midup",
        )

    def forward(
        self,
        noisy_latents: torch.FloatTensor,
        timesteps: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        ref_latents: torch.FloatTensor = None,  # uncond_fwd
    ):
        if ref_latents is not None:
            n = ref_latents.size(0) // noisy_latents.size(0)
            self.referencenet(
                ref_latents,
                torch.zeros_like(timesteps).repeat_interleave(n, dim=0),
                encoder_hidden_states=encoder_hidden_states.repeat_interleave(n, dim=0),
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )

        return model_pred
    
    def clear(self):
        self.reference_control_reader.clear()
        self.reference_control_writer.clear()

def main(args):
    set_seed(args.seed or 42)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    referencenet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    net = ReferenceNet(unet, referencenet)

    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(True)
    #  Some top layer parames of referencenet don't need grad
    for name, param in referencenet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    accelerator.print(weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)


    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    accelerator.print(f"Trainble: {trainable_params/1e6:.1f} / {total_params/1e6:.1f} M ({trainable_params/total_params:.1%})")

    # optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_workers = min(args.dataloader_num_workers, os.cpu_count() - 1)

    train_dataset = AspectDataset(
        metafiles=args.metafiles, tokenizer=tokenizer, 
        proportion_empty_prompts=args.proportion_empty_prompts, proportion_empty_face=args.proportion_empty_face,
    )
    sampler = AspectSampler(
        train_dataset, batch_size = args.train_batch_size, seed = args.seed or 42,
        world_size=accelerator.num_processes, 
        global_rank=accelerator.process_index,
        base_res = (512, 512),
        max_size = 512 * 768,
        dim_range = (256, 768),
    )

    train_dataloader = DataLoader(
        train_dataset, sampler=sampler, batch_size=sampler.batch_size,
        collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers>0
    )

    # Prepare everything with our `accelerator`.
    net, optimizer = accelerator.prepare(net, optimizer)

    global_step = 0
    accelerator.print(f"start training")

    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            if global_step == 0 and accelerator.is_main_process:
                accelerator.print(batch.keys())
            load_data_time = time.perf_counter() - begin
            # Convert images to latent space
            with torch.no_grad():
                input_ids = batch["input_ids"].to(accelerator.device)
                n = int(batch["valid"].reshape(-1, 4).sum(dim=-1).max().item())
                n = np.random.randint(1, max(n, 1) + 1)
                face_pixel_values = batch["face_pixel_values"][:, :n]
                face_pixel_values = face_pixel_values.reshape(-1, *face_pixel_values.shape[-3:])
                face_pixel_values = face_pixel_values.to(accelerator.device, dtype=weight_dtype)
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

                bsz = input_ids.size(0)

                if np.random.random() < args.proportion_empty_prompts:
                    ref_latents = None
                else:
                    ref_latents = vae.encode(face_pixel_values).latent_dist.sample()
                    ref_latents = ref_latents * vae.config.scaling_factor

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.clip_skip <= 1:
                    encoder_hidden_states = text_encoder(input_ids).last_hidden_state
                else:
                    enc_out = text_encoder(input_ids, output_hidden_states=True)
                    encoder_hidden_states = enc_out.hidden_states[-args.clip_skip]
                    encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)


            with accelerator.accumulate(net):

                noise_pred = net(noisy_latents, timesteps, encoder_hidden_states, ref_latents).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(*[x["params"] for x in optimizer.param_groups])
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                accelerator.unwrap_model(net).clear()
                global_step += 1

                if (global_step < 20 or global_step % 500 == 0) and accelerator.is_main_process:
                    print(f"Epoch {epoch}, global step {global_step}, data_time: {load_data_time:.3f}, time: {time.perf_counter() - begin:.3f}, step_loss: {avg_loss:.5f}")

                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet"))
                    accelerator.unwrap_model(referencenet).save_pretrained(os.path.join(save_path, "refnet"))
                    accelerator.print(f"save ckpt to {save_path}")

            begin = time.perf_counter()

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final")
        accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet"))
        accelerator.unwrap_model(referencenet).save_pretrained(os.path.join(save_path, "refnet"))
        accelerator.print(f"save ckpt to {save_path}")


if __name__ == "__main__":
    args = parse_args()

    main(args)    


