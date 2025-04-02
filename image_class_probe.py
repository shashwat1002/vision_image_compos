from scripts.utils.model_init import (
    probe_block,
    probe_model,
    init_diffusion_prior_model,
)
from scripts.convert_image_to_embedding_catdog import (
    create_output_filename,
    convert_raw_to_embeddings,
    convert_raw_to_embeddings_image_encs,
)

import os
import torch

from datasets import load_dataset

from scripts.dataset_script.image_classification_vae_clip import (
    VAEEmbed,
    EncEmberDataset,
)

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger

from scripts.probe_models import ProbeModelWordLabelLightning

import random
import numpy as np

# from transformers import seed_everything as seed_everything_tr

# def seed_everything(seed: int):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     seed_everything_tr(seed)


def create_cache_files(
    dataset_objects,
    device: str,
    output_dir: str,
    model_name="vae",
    model_type="vae",
    force_create: bool = False,
):
    """
    Create the cache files for the dataset and model
    Inputs:
    * dataset_obj: Dataset object: the dataset object
    * model_obj: Model object: the model object
    * model_name: str: the name of the model
    * device: str: the device to use
    * output_dir: str: the output directory
    * model_type: str: the type of the model
    * force_create: bool: whether to force create the cache files
    """

    # create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    splits = ["train", "val", "test"]

    file_names = []

    for i, split in enumerate(splits):
        # create the cache file for the dataset
        if model_type == "vae":
            dataset_cache_file = f"{output_dir}/{model_name}_{split}_dataset_cache.h5"
        else:
            dataset_cache_file = create_output_filename(
                dataset_name="cats_dogs",
                split=split,
                output_dir=output_dir,
                model_name=model_name,
            )
        if not os.path.exists(dataset_cache_file) or force_create:

            if model_type == "vae":
                convert_raw_to_embeddings(
                    dataset_objects[i],
                    output_path=dataset_cache_file,
                    device=device,
                )
            else:
                convert_raw_to_embeddings_image_encs(
                    model_name=model_name,
                    model_type=model_type,
                    dataset_split=dataset_objects[i],
                    split=split,
                    output_path=dataset_cache_file,
                    device=device,
                )
        file_names.append(dataset_cache_file)

    return file_names


def make_dataset_splits(dataset_name):
    # load the shuffled dataset and break into splits
    dataset = load_dataset(dataset_name)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset["train"]

    train_test_split = dataset.train_test_split(test_size=0.2, shuffle=False)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    val_test_split = test_dataset.train_test_split(test_size=0.5, shuffle=False)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    return train_dataset, val_dataset, test_dataset


def make_embed_dataset(
    dataset_objects, file_names, pooling="concat", encoder_type="vae", only_mean=True
):
    if encoder_type == "vae":
        dataset_train = VAEEmbed(
            dataset=dataset_objects[0],
            file_name=file_names[0],
            pooling=pooling,
            only_mean=only_mean,
        )

        dataset_val = VAEEmbed(
            dataset=dataset_objects[1],
            file_name=file_names[1],
            pooling=pooling,
            only_mean=only_mean,
        )

        dataset_test = VAEEmbed(
            dataset=dataset_objects[2],
            file_name=file_names[2],
            pooling=pooling,
            only_mean=only_mean,
        )

    if encoder_type == "enc":
        dataset_train = EncEmberDataset(
            dataset=dataset_objects[0],
            file_name=file_names[0],
        )

        dataset_val = EncEmberDataset(
            dataset=dataset_objects[1],
            file_name=file_names[1],
        )

        dataset_test = EncEmberDataset(
            dataset=dataset_objects[2],
            file_name=file_names[2],
        )

    return dataset_train, dataset_val, dataset_test


def main(
    dataset_name: str,
    output_dir: str,
    device: str,
    force_create: bool,
    batch_size: int = 64,
    hiddens=[],
    model_name="vae",
    model_type="vae",
    only_mean=False,
):
    """
    Main function
    Inputs:
    * dataset_name: str: name of the dataset
    * dataset_file: str: path to the dataset file
    * model_name: str: name of the model
    * model_type: str: type of the model
    * output_dir: str: path to the output directory
    * device: str: device to use
    * force_create: bool: whether to force create the cache files
    """

    # load the dataset
    dataset_objects = make_dataset_splits(dataset_name)

    # create the cache files
    file_names = create_cache_files(
        dataset_objects,
        device,
        output_dir,
        model_name=model_name,
        model_type=model_type,
        force_create=force_create,
    )

    pooling = "concat"

    dataset_train, dataset_val, dataset_test = make_embed_dataset(
        dataset_objects,
        file_names,
        pooling=pooling,
        encoder_type="vae" if model_name == "vae" else "enc",
        only_mean=only_mean,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=False
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False
    )

    # initialize w&b
    wandb_logger = WandbLogger(project="vae-classification")
    wandb_logger.experiment.config.update(
        {
            "dataset_name": dataset_name,
            "output_dir": output_dir,
            "device": device,
            "force_create": force_create,
            "batch_size": batch_size,
            "hiddens": hiddens,
            "model_name": model_name,
            "model_type": model_type,
            "pooling": pooling,
            "only_mean": only_mean,
        }
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        accelerator="cuda",
        logger=wandb_logger,
        callbacks=[early_stop_callback],
    )

    # probe the model
    model = ProbeModelWordLabelLightning(
        input_dim=dataset_train.dim,
        output_dim=dataset_train.dim,
        hidden_dims=hiddens,
        non_linearity="relu",
        lr=1e-4,
    )

    # train probe
    trainer.fit(model, train_dataloader, val_dataloader)

    # test probe
    trainer.test(model, test_dataloader)

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="microsoft/cats_vs_dogs")
    parser.add_argument(
        "--output_dir", type=str, default="/scratch/shashwat.s/embed_cache"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_create", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hiddens", nargs="+", type=int, default=[])
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--model_type", type=str, default="image")
    parser.add_argument("--only_mean", action="store_true")

    args = parser.parse_args()

    main(
        args.dataset_name,
        args.output_dir,
        args.device,
        args.force_create,
        args.batch_size,
        args.hiddens,
        args.model_name,
        args.model_type,
        args.only_mean,
    )
