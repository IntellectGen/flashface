# open FlashFace
Unofficial PyTorch Implementation for [FlashFace](https://arxiv.org/abs/2403.17008): A ReferenceNet absed few shot Identity Personalization.

This project still work in process. Please stay tuned for pretrained model releasing.

## Environment
```
torch>2.0
transformers==4.34.1
diffusers==0.22.1
accelerate==0.23.0
```

## Data
prepare data structure

## Train
```bash
PRETRAINED_MODEL=""
accelerate launch --multi_gpu --main_process_port=21634 --mixed_precision=fp16 train.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL \
    --output_dir output \
    --metafiles data.jsonl \
    --clip_skip 2 \
    --proportion_empty_prompts 0.1 \
    --proportion_empty_face 0.0 \
    --save_steps 20000 \
    --resolution=512 \
    --learning_rate=5e-6 \
    --train_batch_size=8 \
    --dataloader_num_workers=6 \
    --num_train_epochs=20 \
    --mixed_precision=fp16 \
    --seed 42
```

## Inference
```bash
python inference.py
```

## Citation
```
@misc{zhang2024flashface,
      title={FlashFace: Human Image Personalization with High-fidelity Identity Preservation}, 
      author={Shilong Zhang and Lianghua Huang and Xi Chen and Yifei Zhang and Zhi-Fan Wu and Yutong Feng and Wei Wang and Yujun Shen and Yu Liu and Ping Luo},
      year={2024},
      eprint={2403.17008},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
