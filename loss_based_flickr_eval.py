"""
The purpose of this file is to make a script to do the standard winoground evaluation

The script is design to take a pretrained model-name and layer number and then evaluate the model on the winoground dataset.
"""

import argparse
import os
import sys
from typing import List
import torch
from datasets import load_dataset
from scripts.utils.model_init import init_diffusion_prior_model
from tqdm import tqdm

from scripts.dataset_script.winoground_dataset import WinogroundEmbeddingDataset
from scripts.convert_winoground_to_prior_and_clip_embed import (
    convert_raw_to_embeddings,
    create_output_filename,
)
import random
import numpy as np
from scripts.utils.evals import loss_eval
from scripts.utils.feature_extraction import get_openai_clip_image_embeds

# from transformers import seed_everything


# seed everything
def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # seed_everything(seed)


import clip

from PIL import Image


def main(
    model_path,
    split,
    num_samples,
    cond_scale,
    device,
    clip_vanilla=False,
    uncond_norm=False,
):

    dataset = load_dataset(
        "atasoglu/flickr8k-dataset",
        split=f"test",
        data_dir="/scratch/shashwat.s",
    )
    print(len(dataset))
    num_samples = int(len(dataset) * split)

    # randomly sample with a seed
    dataset = dataset.shuffle(seed=69)
    dataset = dataset.select(range(num_samples))

    model_dict = None
    if not clip_vanilla:
        model_dict = init_diffusion_prior_model(model_path, device=device)
        diffusion_prior_trainer = model_dict["trainer"]
    clip_model, processor = clip.load("ViT-L/14")
    clip_model = clip_model.eval().to(device)

    results = []  # each row corresponds to image, column is caption

    with torch.no_grad():
        for i, example in tqdm(enumerate(dataset)):
            image_path = example["image_path"]
            image = Image.open(image_path)
            image_results = []
            uncond_loss = 1.0
            if uncond_norm:
                uncond_loss = loss_eval(
                    caption=" ",
                    diffusion_prior_trainer=diffusion_prior_trainer,
                    clip_model=clip_model,
                    processor=processor,
                    image=image,
                    num_samples=num_samples,
                    device=device,
                    cond_scale=0.0,
                )
            for j, example in enumerate(dataset):
                caption = example["captions"][0]
                # caption = caption.replace(" .", ".")
                caption = caption.strip()
                # print(caption)
                if not clip_vanilla:
                    image_results.append(
                        loss_eval(
                            caption,
                            diffusion_prior_trainer,
                            clip_model,
                            processor,
                            image=image,
                            num_samples=num_samples,
                            device=device,
                            cond_scale=cond_scale,
                        )
                        / uncond_loss
                    )
                else:
                    # score the image caption pair with clip
                    text = clip.tokenize([caption]).to(device)
                    # image = image.resize((224, 224))
                    image = processor(Image.open(image_path)).unsqueeze(0).to(device)
                    # image_features = clip_model.encode_image(image)
                    text_features = clip_model.encode_text(text)
                    image_features = get_openai_clip_image_embeds(
                        clip_model, processor, Image.open(image_path), device=device
                    )
                    image_results.append(
                        (image_features @ text_features.T).cpu().numpy().item()
                    )

            results.append(image_results)

    # each row represents a layer of index i

    # save the results array
    with open(
        f"{cond_scale}_{split}_{uncond_norm}_{clip_vanilla}_n_results.txt", "w"
    ) as f:
        for item in results:
            f.write("%s\n" % item)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model")

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on"
    )

    parser.add_argument(
        "--split", type=float, default=1.0, help="split of the dataset to use"
    )

    parser.add_argument(
        "--num_samples", type=int, default=2, help="number of samples to use"
    )

    # add a boolean flag sweep argument that is false by default
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="If true, will sweep through the cond_scale parameter",
    )

    # add boolean flag to just score with clip
    parser.add_argument(
        "--clip_vanilla",
        action="store_true",
        help="If true, will just score with clip",
    )

    parser.add_argument(
        "--uncond_norm",
        action="store_true",
        help="If true, will normalize the scores by the unconditional score",
    )

    args = parser.parse_args()

    print("Begin...")

    range_ = [i for i in range(1, 10)] + [0] + [-i for i in range(1, 20)]

    results = []
    from tabulate import tabulate
    import gc

    if args.sweep:
        for i in range_:
            cond_scale = 2**i
            print(f"Cond scale: {cond_scale}")
            set_random_seed(69)
            results.append(
                main(
                    model_path=args.model_path,
                    split=args.split,
                    num_samples=args.num_samples,
                    cond_scale=cond_scale,
                    device=args.device,
                    clip_vanilla=args.clip_vanilla,
                    uncond_norm=args.uncond_norm,
                )
            )
            # clear all cuda allocation
            gc.collect()
            torch.cuda.empty_cache()
    else:
        set_random_seed(69)
        main(
            model_path=args.model_path,
            split=args.split,
            num_samples=args.num_samples,
            cond_scale=0.5,
            device=args.device,
            clip_vanilla=args.clip_vanilla,
            uncond_norm=args.uncond_norm,
        )
