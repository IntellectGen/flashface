"""
ref: https://github.com/NovelAI/novelai-aspect-ratio-bucketing/blob/main/bucketmanager.py
"""
from typing import Any, Optional, List, Dict, Tuple, Union, Generic, TypeVar
from dataclasses import dataclass, field
from collections.abc import Hashable
import os
import json
import pickle
from PIL import Image, ImageFile
from PIL.ImageOps import exif_transpose
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np

import torch
from torch.utils.data import Sampler, Dataset, DataLoader
import torchvision.transforms as T 

MASTER_ONLY = True


def load_listdata(filepath):
    _, suffix = os.path.splitext(filepath)
    if suffix == ".pkl":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    elif suffix == ".jsonl":
        with open(filepath, "r") as f:
            data = [json.loads(line.strip()) for line in f.readlines()]
    elif suffix == ".json":
        with open(filepath, "r") as f:
            data = json.loads(f.read())
    else:
        with open(filepath, "r") as f:
            data = f.readlines()
    return data

def save_listdata(data, filepath):
    _, suffix = os.path.splitext(filepath)
    if suffix == ".pkl":
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    elif suffix == ".jsonl":
        with open(filepath, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    elif suffix == ".json":
        with open(filepath, "w") as f:
            f.write(json.dumps(data) + "\n")
    else:
        with open(filepath, "w") as f:
            for d in data:
                f.write(str(d) + "\n")
    return data


Size = Tuple[int, int]
T_id = TypeVar("T_id", bound=Hashable)

@dataclass
class Index:
    value: T_id
    size: Size

@dataclass
class Bucket:
    size: Size
    ids: List[Hashable] = field(default_factory=list)

    def __hash__(self) -> int:
        return self.size.__hash__()

    def __str__(self) -> str:
        return str(self.size)

    @property
    def aspect(self):
        return float(self.size[0]) / float(self.size[1])


class BucketManager:
    def __init__(self, batch_size: int, seed: Optional[int] = None, world_size=1, global_rank=0):
        self.batch_size = batch_size
        self.world_size = world_size
        self.global_rank = global_rank

        self.buckets: Optional[List[Bucket]] = None
        self.idx_to_shape: Dict[T_id, Size] = {}
        self.base_res: Optional[Size] = None
        self.epoch: Optional[dict[Bucket, List[T_id]]] = None
        self.epoch_remainders: Optional[List[T_id]] = None
        self.batch_total = 0
        self.batch_delivered = 0

        self.bucket_prng = np.random.RandomState(seed)
        # separate prng for sharding use for increased thread resilience
        sharding_seed = self.bucket_prng.tomaxint() % (2 ** 32 - 1)
        self.sharding_prng = np.random.RandomState(sharding_seed)

    @property
    def is_main_process(self):
        return self.global_rank == 0

    @property
    def epoch_null(self):
        return self.epoch is None or self.epoch_remainders is None

    @property
    def epoch_empty(self):
        return not (len(self.epoch_remainders) > 0 or len(self.epoch)>0) or self.batch_total == self.batch_delivered

    def gen_buckets(self, base_res=(512, 512), max_size=768 * 512, dim_range=(256, 1024), divisor=64):
        min_dim, max_dim = dim_range
        resolutions = set()

        w = min_dim
        while w * min_dim <= max_size and w <= max_dim:
            h = min_dim
            while w * (h + divisor) <= max_size and (h + divisor) <= max_dim:
                if (w, h) == base_res:
                    resolutions.add(base_res)
                h += divisor
            resolutions.add((w, h))
            w += divisor

        h = min_dim
        while h / min_dim <= max_size and h <= max_dim:
            w = min_dim
            while h * (w + divisor) <= max_size and (w + divisor) <= max_dim:
                w += divisor
            resolutions.add((w, h))
            h += divisor

        self.base_res = base_res
        self.buckets = [Bucket(res) for res in sorted(resolutions)]

        if self.is_main_process or not MASTER_ONLY:
            print(f"rank: {self.global_rank} Bucket sizes: {resolutions}")

    def put_in(self, idx_to_shape: Dict[T_id, Size], max_aspect_error=0.5):
        self.idx_to_shape = idx_to_shape
        aspect_errors = []
        skipped_ids = []

        for id, (w, h) in idx_to_shape.items():
            aspect = float(w) / float(h)
            best_fit_bucket: Bucket = min(self.buckets, key=lambda b: abs(b.aspect - aspect))
            error = abs(best_fit_bucket.aspect - aspect)
            if error < max_aspect_error:
                best_fit_bucket.ids.append(id)
                aspect_errors.append(error)
            else:
                skipped_ids.append(id)

        aspect_errors = np.array(aspect_errors)

        if self.is_main_process or not MASTER_ONLY:
            print(f"rank: {self.global_rank} Aspect Error: mean {np.mean(aspect_errors):.3f}, median {np.median(aspect_errors):.3f}, max {np.max(aspect_errors):.3f}")
            print(f"rank: {self.global_rank} Skipped Entries: {len(skipped_ids)}")
            for bucket in self.buckets:
                print(f"rank: {self.global_rank} Bucket {bucket.size}, aspect {bucket.aspect:.5f}, {len(bucket.ids)} entries")

    def _get_local_ids(self):
        """Select ids of an epoch for this local rank."""
        local_ids = list(self.idx_to_shape.keys())
        index_len = len(local_ids)
        self.sharding_prng.shuffle(local_ids)

        local_ids = local_ids[:index_len - (index_len % (self.batch_size * self.world_size))]
        local_ids = local_ids[self.global_rank::self.world_size]

        index_len = len(local_ids)
        self.batch_total = index_len // self.batch_size
        assert (index_len % self.batch_size == 0)

        local_ids = set(local_ids)
        return local_ids

    def start_epoch(self):
        local_ids = self._get_local_ids()
        epoch = {}
        epoch_remainders = []

        for bucket in self.buckets:
            if len(bucket.ids) == 0:
                continue

            chosen_ids = [id for id in bucket.ids if id in local_ids]
            self.bucket_prng.shuffle(chosen_ids)

            remainder = len(chosen_ids) % self.batch_size
            if remainder != 0:
                chosen_ids, remainders = chosen_ids[remainder:], chosen_ids[:remainder]
                epoch_remainders.extend(remainders)

            if len(chosen_ids) == 0:
                continue

            epoch[bucket] = chosen_ids

        self.epoch = epoch
        self.epoch_remainders = epoch_remainders
        self.batch_delivered = 0

        if self.is_main_process or not MASTER_ONLY:
            print(f"rank: {self.global_rank} Correct item: {sum(len(ids) for ids in epoch.values())} / {len(local_ids)}")

    def get_batch(self):
        if self.epoch_null:
            raise Exception("No epoch")

        resolution = self.base_res
        found_batch = False
        batch_buckets: List[T_id] = []
        chosen_bucket: Optional[Bucket] = None

        while not found_batch:
            buckets: List[Union[Bucket, str]] = list(self.epoch.keys())

            bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in buckets]

            if len(self.epoch_remainders) >= self.batch_size:
                buckets.append("left_over")
                bucket_probs.append(len(self.epoch_remainders))

            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            # Buckets with more images get more weight
            bucket_probs /= bucket_probs.sum()

            chosen_bucket = self.bucket_prng.choice(buckets, 1, p=bucket_probs)[0] if len(self.epoch)>0 else "left_over"

            if chosen_bucket == "left_over":
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                chosen_ids = self.epoch_remainders
                self.bucket_prng.shuffle(chosen_ids)
                self.epoch_remainders, batch_buckets = chosen_ids[self.batch_size:], chosen_ids[:self.batch_size]
                found_batch = True
            else:
                chosen_ids = self.epoch[chosen_bucket]
                if len(chosen_ids) >= self.batch_size:
                    # return bucket batch and resolution
                    self.epoch[chosen_bucket], batch_buckets = chosen_ids[self.batch_size:], chosen_ids[:self.batch_size]
                    resolution = chosen_bucket.size
                    found_batch = True
                    if len(self.epoch[chosen_bucket]) == 0:
                        del self.epoch[chosen_bucket]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.epoch_remainders.extend(chosen_ids)
                    del self.epoch[chosen_bucket]

            assert (found_batch or len(self.epoch_remainders) >= self.batch_size or len(self.epoch)>0)

        # print("Bucket probs: ", ", ".join(map(lambda x: f"{x:.2f}%", list(bucket_probs * 100))))
        # print(f"Chosen bucket: {chosen_bucket}")
        # print("Batch data", batch_buckets)

        self.batch_delivered += 1
        return batch_buckets, resolution

    def generator(self):
        if self.epoch_null or self.epoch_empty:
            self.start_epoch()

        while not self.epoch_empty:
            yield self.get_batch()


class AspectSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        world_size: int,
        global_rank: int,
        seed: int = 42,
        max_aspect_error: float = 0.5,
        base_res = (640, 640),
        max_size = 640 * 640,
        dim_range = (320, 960),
    ):
        super().__init__(dataset)

        bucket_manager = BucketManager(batch_size, seed, world_size, global_rank)
        bucket_manager.gen_buckets(base_res=base_res, max_size=max_size, dim_range=dim_range, divisor=64)
        bucket_manager.put_in(dataset.idx_to_shape, max_aspect_error)

        self.bucket_manager = bucket_manager
        self._world_size = world_size
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        for batch, size in self.bucket_manager.generator():
            yield from (Index(index, size) for index in batch)

    def __len__(self):
        if self.bucket_manager.epoch_null:
            self.bucket_manager.start_epoch()

        return self.bucket_manager.batch_total * self._batch_size


class AspectDataset(Dataset):
    def __init__(self, metafiles: List[str] = ["metadata.jsonl"], tokenizer=None, proportion_empty_prompts=0.1, proportion_empty_face=0.5, max_face_num=4):
        assert tokenizer is not None, "tokenizer is None"
        self.max_face_num = max_face_num
        self.proportion_empty_prompts = proportion_empty_prompts
        self.proportion_empty_face = proportion_empty_face
        self.data: List[Dict[str, Any]] = []
        for metafile in metafiles:
            data = load_listdata(metafile)
            data = [v for v in data if max(v['size']) > 256]
            self.data.extend(data)
        self.idx_to_shape = {k: d["size"] for k, d in enumerate(self.data)}
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.5), std=(0.5)),
        ])
        self.tokenizer = tokenizer
        self.face_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize((256, 256), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=(0.5), std=(0.5)),
        ])

        print(f"datasize: {len(self.data)} -> {len(self.idx_to_shape)}")

    def _read_image(self, filepath: str) -> Image.Image:
        image = Image.open(filepath)
        assert isinstance(image, Image.Image)
        image = exif_transpose(image)
        image = image.convert("RGB")
        return image

    def __len__(self):
        return len(self.idx_to_shape)

    def __getitem__(self, index: Index):
        idx = index.value
        width, height = index.size
        metadata = self.data[idx]
        caption = metadata["caption"]
        raw_width, raw_height = metadata["size"]
        path = metadata["path"]
        ref = metadata.get("ref", [])
        ref_faces = []


        try:
            image = self._read_image(path)
            w, h = image.size
            aspect_target = width / height
            aspect_image = w / h
            if aspect_image > aspect_target:
                h = height
                w = round(height * aspect_image)
            elif aspect_image < aspect_target:
                w = width
                h = round(width / aspect_image)
            else:
                w = width
                h = height

            left = 0 if w<=width else np.random.randint(0, w - width)
            top  = 0 if h<=height else np.random.randint(0, h - height)
            right = left + width
            bottom = top + height

            image = image.resize((w, h), resample=Image.Resampling.LANCZOS).crop((left, top, right, bottom))

            if len(ref) > 0 and np.random.random() > self.proportion_empty_face:
                h = min(len(ref), self.max_face_num)
                face_paths = np.random.choice(ref, np.random.randint(1, h+1))
                for face_path in face_paths:
                    try:
                        ref_face = self._read_image(face_path)
                        ref_faces.append(ref_face)
                    except Exception as e:
                        print(e)

        except Exception as e:
            print(metadata['path'], e)
            image = Image.new("RGB", (width, height), color=(0, 0, 0))
            caption = ""

        pixel_values = self.transform(image)
        face_pixel_values = torch.zeros(self.max_face_num, 3, 256, 256)
        if len(ref_faces) > 0:
            face_pixel_values[:len(ref_faces)] = torch.stack([self.ref_face_transform(ref_face) for ref_face in ref_faces], dim=0)

        if np.random.random() < self.proportion_empty_prompts:
            caption = ""
        input_ids = self.tokenizer(caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

        sample = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "face_pixel_values": face_pixel_values,
        }

        return sample


def collate_fn(examples):

    input_ids = torch.cat([example["input_ids"] for example in examples]).long()

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.contiguous().float()

    face_pixel_values = torch.stack([example["face_pixel_values"] for example in examples], dim=0)
    face_pixel_values = face_pixel_values.contiguous().float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "face_pixel_values": face_pixel_values,
    }

    return batch


if __name__ == "__main__":
    from tqdm import tqdm
    from torchvision.utils import make_grid, save_image
    from accelerate import Accelerator
    from transformers import CLIPTokenizer
    from parsenet import FaceSeg

    accelerator = Accelerator()

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

    dataset = AspectDataset(
        metafiles=[
            "data.jsonl"
        ], tokenizer=tokenizer, 
        proportion_empty_prompts=0.1, proportion_empty_face=0.0,
    )
    sampler = AspectSampler(
        dataset, batch_size=8, seed=40,
        world_size=accelerator.num_processes, 
        global_rank=accelerator.process_index,
        base_res = (512, 512),
        max_size = 512 * 768,
        dim_range = (256, 768),
    )

    num_workers = 0
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=sampler.batch_size,
        collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers>0
    )


    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), ncols=80, disable=True):

        if idx < 40 and idx % 10 == 0:
            print(idx, {k: tuple(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)})
            save_image(make_grid(batch["pixel_values"], nrow=4, padding = 4, normalize=True, value_range=(-1, 1.0)), f"sample_image_{idx}.jpg")
            face_pixel_values = batch["face_pixel_values"]
            save_image(make_grid(face_pixel_values.reshape(-1, *face_pixel_values.shape[-3:]), nrow=face_pixel_values.size(1), padding = 4, normalize=True, value_range=(-1, 1.0)), f"sample_face_{idx}.jpg")

        if idx > 40:
            break
