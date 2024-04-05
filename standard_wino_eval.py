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
from scripts.utils.model_init import init_subject_model
from tqdm import tqdm


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


def distance_eval(e1: torch.Tensor, e2: torch.Tensor, metric: str) -> bool:
    if metric == "cosine":
        return 1 - torch.nn.functional.cosine_similarity(e1, e2)
    elif metric == "euclidean":
        # print(torch.norm(e1 - e2))
        return torch.norm(e1 - e2)


def get_embedding(model_dict, c1, c2, i1, i2, pooling: str = "pooler"):
    processor = model_dict["processor"]
    tokenizer = model_dict["tokenizer"]
    model_text = model_dict["model_text"]
    model_image = model_dict["model"].vision_model
    device = model_dict["model"].device
    model = model_dict["model"]

    input_image_1 = processor(images=i1, return_tensors="pt", padding=True).to(
        device=device
    )
    input_image_2 = processor(images=i2, return_tensors="pt", padding=True).to(
        device=device
    )
    input_text_1 = tokenizer(c1, return_tensors="pt", padding=True).to(device=device)
    input_text_2 = tokenizer(c2, return_tensors="pt", padding=True).to(device=device)
    eos_token = model_text.config.eos_token_id

    # get the index of eos_token_for_both
    eos_token_index_1 = (input_text_1["input_ids"] == eos_token).int().argmax(dim=-1)
    eos_token_index_2 = (input_text_2["input_ids"] == eos_token).int().argmax(dim=-1)
    print(eos_token_index_1, eos_token_index_2)

    text_projection = model.text_projection
    image_projection = model.visual_projection

    with torch.no_grad():
        # print(model_image(**input_image_1).last_hidden_state.shape)
        image_1_features = image_projection(model_image(**input_image_1).pooler_output)
        image_2_features = image_projection(model_image(**input_image_2).pooler_output)
        print(image_2_features.shape)
        # image_2_features
        text_1_features, text_2_features = [], []
        if pooling == "pooler":
            text_1_features = [
                text_projection(model_text.final_layer_norm(embds))[
                    :, eos_token_index_1, :
                ].squeeze()
                for embds in model_text(
                    **input_text_1, output_hidden_states=True
                ).hidden_states
            ]
            text_2_features = [
                text_projection(model_text.final_layer_norm(embds))[
                    :, eos_token_index_2, :
                ].squeeze()
                for embds in model_text(
                    **input_text_2, output_hidden_states=True
                ).hidden_states
            ]
        if pooling == "mean":
            text_1_features = [
                text_projection(model_text.final_layer_norm(embds))
                .mean(dim=1)
                .squeeze()
                for embds in model_text(
                    **input_text_1, output_hidden_states=True
                ).hidden_states
            ]
            text_2_features = [
                text_projection(model_text.final_layer_norm(embds))
                .mean(dim=1)
                .squeeze()
                for embds in model_text(
                    **input_text_2, output_hidden_states=True
                ).hidden_states
            ]
    return text_1_features, text_2_features, image_1_features, image_2_features


def main(args):

    dataset = load_dataset("facebook/winoground")["test"]
    model = args.model
    metric = args.metric
    device = args.device

    model_dict = init_subject_model(model, "clip", device=device)

    total = 0
    text_eval_list = [0 for _ in range(model_dict["config_text"].num_hidden_layers + 1)]
    image_eval_list = [
        0 for _ in range(model_dict["config_text"].num_hidden_layers + 1)
    ]
    group_eval_list = [
        0 for _ in range(model_dict["config_text"].num_hidden_layers + 1)
    ]

    for example in tqdm(dataset):
        c1 = example["caption_0"]
        c2 = example["caption_1"]
        i1 = example["image_0"].convert("RGB")
        i2 = example["image_1"].convert("RGB")

        c1_embed_list, c2_embed_list, i1_embed, i2_embed = get_embedding(
            model_dict, c1, c2, i1, i2
        )

        for i in range(model_dict["config_text"].num_hidden_layers + 1):
            if text_eval(
                c1_embed_list[i], c2_embed_list[i], i1_embed, i2_embed, metric
            ):
                # print("hi")
                text_eval_list[i] += 1

            if group_eval(
                c1_embed_list[i], c2_embed_list[i], i1_embed, i2_embed, metric
            ):
                group_eval_list[i] += 1

            if image_eval(
                c1_embed_list[i], c2_embed_list[i], i1_embed, i2_embed, metric
            ):
                image_eval_list[i] += 1

        total += 1

    image_eval_accuracy = [x / total for x in image_eval_list]
    text_eval_accuracy = [x / total for x in text_eval_list]
    group_eval_accuracy = [x / total for x in group_eval_list]

    # print as table
    from tabulate import tabulate

    # each row represents a layer of index i
    rows = []
    for i in range(model_dict["config_text"].num_hidden_layers + 1):
        rows.append(
            [i, text_eval_accuracy[i], image_eval_accuracy[i], group_eval_accuracy[i]]
        )
    print(tabulate(rows, headers=["Layer", "Text", "Image", "Group"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric to use for evaluation",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on"
    )
    args = parser.parse_args()
    main(args)
