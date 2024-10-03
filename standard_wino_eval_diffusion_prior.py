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
from scripts.utils.feature_extraction import get_embedding_wino_eval
from scripts.dataset_script.winoground_dataset import WinogroundEmbeddingDataset
from scripts.convert_winoground_to_prior_and_clip_embed import (
    convert_raw_to_embeddings,
    create_output_filename,
)
import random
import numpy as np

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


def create_cache_files(
    dataset_name: str,
    path: str,
    device: str,
    output_dir: str,
    force_create: bool = False,
    predict: bool = True,
    split: float = 1.0,
    agg: str = "default",
    num_samples: int = 2,
):
    print("value of predict", predict)
    text_cache, img_cache = create_output_filename(
        input_path=dataset_name,
        output_dir=output_dir,
        model_path=path,
        predicted=predict,
        split=split,
        agg=agg,
    )

    # checking files that exist
    actual_files_to_process = []
    if not force_create:
        for i, output_file in enumerate([text_cache, img_cache]):
            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping.")
            else:
                actual_files_to_process = [text_cache, img_cache]
    else:
        actual_files_to_process = [text_cache, img_cache]
    split = 100 * split
    dataset_split = load_dataset(dataset_name, split=f"test[:{int(split)}%]")
    if len(actual_files_to_process) != 0:
        convert_raw_to_embeddings(
            dataset_split,
            model_path=path,
            output_paths=(text_cache, img_cache),
            device=device,
            predicted=predict,
            num_samples=num_samples,
            agg=agg,
        )
    print("Embeddings created successfully")

    return {
        "text": text_cache,
        "img": img_cache,
    }


def text_eval(
    c1_embed: torch.Tensor,
    c2_embed: torch.Tensor,
    i1_embed: torch.Tensor,
    i2_embed: torch.Tensor,
    distance_metric: str,
) -> bool:
    return distance_eval(c1_embed, i1_embed, distance_metric) < distance_eval(
        c2_embed, i1_embed, distance_metric
    ) and distance_eval(c2_embed, i2_embed, distance_metric) < distance_eval(
        c1_embed, i2_embed, distance_metric
    )


def image_eval(
    c1_embed: torch.Tensor,
    c2_embed: torch.Tensor,
    i1_embed: torch.Tensor,
    i2_embed: torch.Tensor,
    distance_metric: str,
) -> bool:
    return distance_eval(i1_embed, c1_embed, distance_metric) < distance_eval(
        i2_embed, c1_embed, distance_metric
    ) and distance_eval(i2_embed, c2_embed, distance_metric) < distance_eval(
        i1_embed, c2_embed, distance_metric
    )


def group_eval(
    c1_embed: torch.Tensor,
    c2_embed: torch.Tensor,
    i1_embed: torch.Tensor,
    i2_embed: torch.Tensor,
    distane_metric: str,
) -> bool:
    return text_eval(
        c1_embed, c2_embed, i1_embed, i2_embed, distance_metric=distane_metric
    ) and image_eval(
        c1_embed, c2_embed, i1_embed, i2_embed, distance_metric=distane_metric
    )


def all_distance_measurements(
    c1_embed: torch.Tensor,
    c2_embed: torch.Tensor,
    i1_embed: torch.Tensor,
    i2_embed: torch.Tensor,
    distance_metric: str,
) -> dict[str, float]:

    return {
        "c1, c2": distance_eval(c1_embed, c2_embed, distance_metric).item(),
        "c1, i1": distance_eval(c1_embed, i1_embed, distance_metric).item(),
        "c1, i2": distance_eval(c1_embed, i2_embed, distance_metric).item(),
        "c2, i1": distance_eval(c2_embed, i1_embed, distance_metric).item(),
        "c2, i2": distance_eval(c2_embed, i2_embed, distance_metric).item(),
        "i1, i2": distance_eval(i1_embed, i2_embed, distance_metric).item(),
    }


def distance_eval(e1: torch.Tensor, e2: torch.Tensor, metric: str) -> bool:
    if metric == "cosine":
        if e1.shape[-2] != 1 or e2.shape[-2] != 1:
            raise NotImplementedError()
        return 1 - torch.nn.functional.cosine_similarity(e1, e2)
    elif metric == "euclidean":
        # print(torch.norm(e1 - e2))
        if e1.shape[-2] == 1 and e2.shape[-2] == 1:
            return torch.norm(e1 - e2)
        else:
            # average of distance if one of them is 1 dimensional
            return (e1 - e2).norm(dim=-1).mean()


def make_row_log(
    dataset_index,
    metric,
    c1_embed,
    c2_embed,
    i1_embed,
    i2_embed,
):
    all_distances = all_distance_measurements(
        c1_embed, c2_embed, i1_embed, i2_embed, metric
    )
    row = {
        "dataset_index": dataset_index,
    }

    row.update(all_distances)

    return row


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


def main(args):

    split = args.split
    dataset = load_dataset("facebook/winoground", split=f"test[:{int(100*split)}%]")
    model_path = args.model_path
    metric = args.metric
    device = args.device
    cache_dir = args.cache_dir
    predict = args.predict
    num_samples = args.num_samples
    agg = args.agg

    cache_files = create_cache_files(
        "facebook/winoground",
        model_path,
        device,
        cache_dir,
        predict=predict,
        split=split,
        agg=agg,
        num_samples=num_samples,
    )
    total = 0

    dataset = WinogroundEmbeddingDataset(
        cache_files["text"], cache_files["img"], layer_number=0
    )

    text_eval_n = 0
    image_eval_n = 0
    group_eval_n = 0

    distance_log = []

    for i, (text_sam, image_sam) in tqdm(enumerate(dataset)):
        c1 = text_sam[0, :].unsqueeze(0)
        c2 = text_sam[1, :].unsqueeze(0)
        i1 = image_sam[0, :].unsqueeze(0)
        i2 = image_sam[1, :].unsqueeze(0)

        if text_eval(c1, c2, i1, i2, metric):
            text_eval_n += 1

        if group_eval(c1, c2, i1, i2, metric):
            group_eval_n += 1

        if image_eval(c1, c2, i1, i2, metric):
            image_eval_n += 1

        distance_log.append(make_row_log(i, metric, c1, c2, i1, i2))

    total = len(dataset)

    image_eval_accuracy = image_eval_n / total
    text_eval_accuracy = text_eval_n / total
    group_eval_accuracy = group_eval_n / total

    # print as table
    from tabulate import tabulate

    # each row represents a layer of index i
    rows = [[0, text_eval_accuracy, image_eval_accuracy, group_eval_accuracy, total]]

    print(tabulate(rows, headers=["Layer", "Text", "Image", "Group", "total"]))

    import pandas as pd

    df = pd.DataFrame(distance_log)
    df.to_csv(
        log_filename(
            dataset_name="winoground",
            model_path=model_path,
            metric=metric,
            split=split,
            agg=agg,
            num_samples=num_samples,
            predict=predict,
        ),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model")

    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric to use for evaluation",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on"
    )

    parser.add_argument(
        "--cache_dir", type=str, help="Directory to store the cache files"
    )

    parser.add_argument(
        "--predict", type=bool, help="whether to use predicted embeddings"
    )

    parser.add_argument(
        "--split", type=float, default=1.0, help="split of the dataset to use"
    )

    parser.add_argument(
        "--num_samples", type=int, default=2, help="number of samples to use"
    )

    parser.add_argument(
        "--agg", type=str, default="default", help="aggregation method to use"
    )

    args = parser.parse_args()

    print("Begin...")

    set_random_seed(42)

    main(args)
