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


def all_pair_wise_loss_eval(
    caption1,
    caption2,
    diffusion_prior_trainer,
    clip_model,
    processor,
    image_embed1=None,
    image1=None,
    image_embed2=None,
    image2=None,
    num_samples=10,
    device="cpu",
    cond_scale=None,
):
    """
    Evaluate the loss of the model
    :caption: the caption to evaluate (in string)
    :diffusion_prior_trainer: the model to evaluate
    :clip: the clip model
    :image_embed1 / 2: the image embedding to evaluate (send this either this or image)
    :image1 / 2: the image to evaluate (send this either this or image_embed)
    """

    c1i1 = loss_eval(
        caption1,
        diffusion_prior_trainer,
        clip_model,
        processor,
        image_embed=image_embed1,
        image=image1,
        num_samples=num_samples,
        device=device,
        cond_scale=cond_scale,
    )

    c1i2 = loss_eval(
        caption1,
        diffusion_prior_trainer,
        clip_model,
        processor,
        image_embed=image_embed2,
        image=image2,
        num_samples=num_samples,
        device=device,
        cond_scale=cond_scale,
    )

    c2i1 = loss_eval(
        caption2,
        diffusion_prior_trainer,
        clip_model,
        processor,
        image_embed=image_embed1,
        image=image1,
        num_samples=num_samples,
        device=device,
        cond_scale=cond_scale,
    )

    c2i2 = loss_eval(
        caption2,
        diffusion_prior_trainer,
        clip_model,
        processor,
        image_embed=image_embed2,
        image=image2,
        num_samples=num_samples,
        device=device,
        cond_scale=cond_scale,
    )

    return {
        "c1i1": c1i1,
        "c1i2": c1i2,
        "c2i1": c2i1,
        "c2i2": c2i2,
    }


def text_eval(
    c1i1,
    c2i1,
    c1i2,
    c2i2,
):
    return c1i1 < c2i1 and c2i2 < c1i2


def image_eval(
    c1i1,
    c2i1,
    c1i2,
    c2i2,
):
    return c1i1 < c1i2 and c2i2 < c2i1


def group_eval(
    c1i1,
    c2i1,
    c1i2,
    c2i2,
):
    return text_eval(
        c1i1,
        c2i1,
        c1i2,
        c2i2,
    ) and image_eval(
        c1i1,
        c2i1,
        c1i2,
        c2i2,
    )


# def make_row_log(
#     dataset_index,
#     metric,
#     c1_embed,
#     c2_embed,
#     i1_embed,
#     i2_embed,
# ):
#     all_distances = all_distance_measurements(
#         c1_embed, c2_embed, i1_embed, i2_embed, metric
#     )
#     row = {
#         "dataset_index": dataset_index,
#     }

#     row.update(all_distances)

#     return row


def log_filename(
    dataset_name,
    model_path,
    metric,
    split,
    agg,
    num_samples,
    predict,
):
    dataset_name = dataset_name.replace("/", "_")
    model_path = model_path.split("/")[-1]
    return f"{dataset_name}_{model_path}_{metric}_{split}_{agg}_{num_samples}_{predict}.csv"


def main(
    model_path,
    split,
    num_samples,
    cond_scale,
    device,
):

    dataset = load_dataset("facebook/winoground", split=f"test[:{int(100*split)}%]")

    total = 0

    text_eval_n = 0
    image_eval_n = 0
    group_eval_n = 0

    model_dict = init_diffusion_prior_model(model_path, device=device)
    diffusion_prior_trainer = model_dict["trainer"]
    clip_model, processor = clip.load("ViT-L/14")
    clip_model = clip_model.eval().to(device)

    with torch.no_grad():
        for i, example in tqdm(enumerate(dataset)):
            c1 = example["caption_0"]
            c2 = example["caption_1"]
            i1 = example["image_0"]
            i2 = example["image_1"]

            # get pairwise loss
            all_distances = all_pair_wise_loss_eval(
                c1,
                c2,
                diffusion_prior_trainer,
                clip_model,
                processor,
                # image_embed1=i1,
                image1=i1,
                # image_embed2=i2,
                image2=i2,
                num_samples=num_samples,
                device=device,
                cond_scale=cond_scale,
            )

            # evaluate
            text_eval_n += text_eval(
                all_distances["c1i1"],
                all_distances["c2i1"],
                all_distances["c1i2"],
                all_distances["c2i2"],
            )

            image_eval_n += image_eval(
                all_distances["c1i1"],
                all_distances["c2i1"],
                all_distances["c1i2"],
                all_distances["c2i2"],
            )

            group_eval_n += group_eval(
                all_distances["c1i1"],
                all_distances["c2i1"],
                all_distances["c1i2"],
                all_distances["c2i2"],
            )

    total = len(dataset)

    image_eval_accuracy = image_eval_n / total
    text_eval_accuracy = text_eval_n / total
    group_eval_accuracy = group_eval_n / total

    # print as table
    from tabulate import tabulate

    # each row represents a layer of index i
    rows = [
        [
            0,
            text_eval_accuracy,
            image_eval_accuracy,
            group_eval_accuracy,
            total,
            cond_scale,
        ]
    ]

    print(
        tabulate(
            rows, headers=["Layer", "Text", "Image", "Group", "total", "cond_scale"]
        )
    )

    model_dict = None
    return rows

    # import pandas as pd

    # df = pd.DataFrame(distance_log)
    # df.to_csv(
    #     log_filename(
    #         dataset_name="winoground",
    #         model_path=model_path,
    #         metric=metric,
    #         split=split,
    #         agg=agg,
    #         num_samples=num_samples,
    #         predict=predict,
    #     ),
    #     index=False,
    # )


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

    args = parser.parse_args()

    print("Begin...")

    range_ = [-i for i in range(1, 20)] + [0] + [i for i in range(1, 20)]

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
                )
            )
            # clear all cuda allocation
            gc.collect()
            torch.cuda.empty_cache()
        tabulate(results)
    else:
        set_random_seed(69)
        main(
            model_path=args.model_path,
            split=args.split,
            num_samples=args.num_samples,
            cond_scale=0.5,
            device=args.device,
        )
