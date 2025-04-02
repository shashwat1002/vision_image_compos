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
import h5py

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
        # diffusion_prior_trainer = model_dict["trainer"]
        diffusion_prior = model_dict["diffusion_prior"]
    clip_model, processor = clip.load("ViT-L/14")
    clip_model = clip_model.eval().to(device)

    results = []  # each row corresponds to image, column is caption

    text_embed_file = f"/scratch/shashwat.s/flikr8k_text_embeds_{split}.h5"
    image_embed_file = f"/scratch/shashwat.s/flikr8k_image_embeds_{split}.h5"
    predicted_image_files = f"/scratch/shashwat.s/flikr8k_predicted_images_{split}.h5"

    with torch.no_grad():
        # open the h5 files
        text_embed_h5 = h5py.File(text_embed_file, "w")
        image_embed_h5 = h5py.File(image_embed_file, "w")
        predicted_image_h5 = h5py.File(predicted_image_files, "w")

        for i, example in tqdm(enumerate(dataset)):
            image_path = example["image_path"]
            image = Image.open(image_path)
            image_embed = get_openai_clip_image_embeds(
                model=clip_model, preprocess=processor, i=image, device=device
            )
            text_embed = clip_model.encode_text(
                clip.tokenize(example["captions"][0]).to(device, dtype=torch.long)
            )
            # sampe from prior
            predicted_image = diffusion_prior.sample(
                text=clip.tokenize(example["captions"][0]).to(device, dtype=torch.long),
                cond_scale=cond_scale,
                num_samples_per_batch=num_samples,
            )
            # save the embeddings
            text_embed_h5.create_dataset(f"{i}", data=text_embed.cpu().numpy())
            image_embed_h5.create_dataset(f"{i}", data=image_embed.cpu().numpy())
            predicted_image_h5.create_dataset(
                f"{i}", data=predicted_image.cpu().numpy()
            )

        text_embed_h5.close()
        image_embed_h5.close()
        predicted_image_h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the model",
        default="/scratch/shashwat.s/diff_prior_ema855M.pth",
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on"
    )

    parser.add_argument(
        "--split", type=float, default=1.0, help="split of the dataset to use"
    )

    parser.add_argument(
        "--num_samples", type=int, default=2, help="number of samples to use"
    )

    # add boolean flag to just score with clip
    parser.add_argument(
        "--clip_vanilla",
        action="store_true",
        help="If true, will just score with clip",
    )

    args = parser.parse_args()

    print("Begin...")

    set_random_seed(69)
    main(
        model_path=args.model_path,
        split=args.split,
        num_samples=args.num_samples,
        cond_scale=0.5,
        device=args.device,
        clip_vanilla=args.clip_vanilla,
    )
