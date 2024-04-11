import argparse
import os
import sys
from typing import List
import torch
from datasets import load_dataset
from scripts.utils.model_init import init_subject_model
from tqdm import tqdm
from scripts.utils.feature_extraction import get_embedding_wino_eval
from scripts.dataset_script.winoground_dataset import WinogroundEmbeddingDataset
from scripts.convert_winoground_to_embeddings import (
    convert_raw_to_embeddings,
    create_output_filename,
)
from scripts.probe_models import (
    ProbeSimilarityOrdering,
    ProbeSimilarityOrderingWinnogroundStyle,
)

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger


def create_cache_files(
    dataset_name: str,
    model_name: str,
    model_type: str,
    device: str,
    output_dir: str,
    proj: bool = True,
    pool_strat: str = "pooler",
    force_create: bool = False,
):

    text_cache, img_cache = create_output_filename(
        input_path=dataset_name,
        output_dir=output_dir,
        model_name=model_name,
        pool_strat=pool_strat,
        proj=proj,
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

    dataset_split = load_dataset(dataset_name)["test"]
    if len(actual_files_to_process) != 0:
        convert_raw_to_embeddings(
            dataset_split,
            model_name=model_name,
            model_type=model_type,
            output_paths=(text_cache, img_cache),
            device=device,
            proj=proj,
        )
    print("Embeddings created successfully")

    return {
        "text": text_cache,
        "img": img_cache,
    }


def name_of_run(
    model_name: str,
    hiddens: List[int] = [],
    probe_layer: int = -1,
    proj: bool = True,
    symmetric: bool = True,
):
    model_name = model_name.replace("/", "_")
    return f"{model_name}_wino_{'_'.join([str(x) for x in hiddens])}_layer_{probe_layer}_proj_{proj}_symmetric_{symmetric}"


def run_exp(args):
    dataset = load_dataset("facebook/winoground")["test"]

    model = args.model
    device = args.device
    cache_dir = args.cache_dir
    hiddens = args.hiddens
    layer_number = args.layer_number
    proj = args.proj
    print(f"proj: {proj}")

    model_dict = init_subject_model(model, "clip", device=device)

    # turn a subset of the dataset into another dataset
    dataset_train = dataset.select(range(100))
    dataset_val = dataset.select(range(100, 200))
    dataset_test = dataset.select(range(200, 400))

    # create the cache files
    create_cache_files(
        dataset_name="facebook/winoground",
        model_name=model,
        model_type="clip",
        device=device,
        output_dir=cache_dir,
        force_create=False,
        proj=proj,
    )

    # create the datasets
    text_cache_path, img_cache_path = create_output_filename(
        input_path="facebook/winoground",
        output_dir=cache_dir,
        model_name=model,
        pool_strat="pooler",
        proj=proj,
    )

    dataset = WinogroundEmbeddingDataset(text_cache_path, img_cache_path, layer_number)
    # print(dataset[0].shape)

    train_dataset = WinogroundEmbeddingDataset(
        text_cache_path, img_cache_path, layer_number
    )
    val_dataset = WinogroundEmbeddingDataset(
        text_cache_path, img_cache_path, layer_number
    )
    test_dataset = WinogroundEmbeddingDataset(
        text_cache_path, img_cache_path, layer_number
    )

    train_dataset.single_layer_text_list = train_dataset.single_layer_text_list[:100]
    val_dataset.single_layer_text_list = val_dataset.single_layer_text_list[100:200]
    test_dataset.single_layer_text_list = test_dataset.single_layer_text_list[200:]

    train_dataset.image_feature_list = train_dataset.image_feature_list[:100]
    val_dataset.image_feature_list = val_dataset.image_feature_list[100:200]
    test_dataset.image_feature_list = test_dataset.image_feature_list[200:]

    input_dim_c = train_dataset[0][0].shape[-1]
    input_dim_v = train_dataset[0][1].shape[-1]
    output_dim = input_dim_c
    if args.output_dim is not None:
        output_dim = args.output_dim

    # print(train_dataset[0].shape)

    # make the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False
    )
    # print(next(iter(train_loader)).shape)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    symmetric = False

    # run the experiment
    pl_probe = ProbeSimilarityOrderingWinnogroundStyle(
        input_dim_c=input_dim_c,
        input_dim_v=input_dim_v,
        output_dim=output_dim,
        hidden_dims=hiddens,
        non_linearity="relu",
        symmetric=symmetric,
    )

    # Initialize w&b
    run_name = name_of_run(model, hiddens, layer_number, symmetric=symmetric, proj=proj)
    wandb_logger = WandbLogger(name=run_name, project="winoground_probe")

    # Initialize the trainer
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    trainer = Trainer(
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[early_stopping],
        accelerator=device,
    )

    trainer.fit(pl_probe, train_loader, val_loader)
    trainer.test(pl_probe, test_loader)

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--hiddens", type=int, nargs="+", default=[])
    parser.add_argument("--layer_number", type=int, default=-1)

    parser.add_argument("--proj", action="store_true")
    parser.add_argument("--output_dim", type=int, default=None)

    args = parser.parse_args()
    if args.layer_number == -2:
        for i in range(25):
            args.layer_number = i
            run_exp(args)
    else:
        run_exp(args)
