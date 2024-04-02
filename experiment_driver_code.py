from scripts.utils.model_init import init_subject_model
from scripts.convert_raw_to_embeddings import (
    create_output_filename,
    convert_raw_to_embeddings,
)
import os
from scripts.dataset_script.pos_dataset import EmbeddingsDatasetPos
from scripts.dataset_script.dep_parse_dataset import DepParseDataset
from typing import *
from torch.utils.data import DataLoader
from scripts.utils.data_utils import custom_embedding_padding
from scripts.probe_models import (
    ProbeModelWordLabelLightning,
    ProbeWordPairLabelModelLightning,
)

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger


def create_cache_files(
    train_input: str,
    dev_input: str,
    test_input: str,
    model_name: str,
    model_type: str,
    device: str,
    output_dir: str,
    force_create: bool = False,
):
    """
    Creates embedding cache for all the files.
    First checks if the file exists (unless force flag is on)
    :param train_input: Path to the training file
    :param dev_input: Path to the dev file
    :param test_input: Path to the test file
    :param model_name: Name of the model
    :param model_type: Type of the model
    :param device: Device to use for embeddings
    :param output_dir: Directory to save the embeddings
    :param force_create: Whether to force create the embeddings
    """

    files = [train_input, dev_input, test_input]
    output_file_names = [
        create_output_filename(
            input_path=filename, output_dir=output_dir, model_name=model_name
        )
        for filename in files
    ]

    # checking files that exist
    actual_files_to_process = []
    if not force_create:
        for i, output_file in enumerate(output_file_names):
            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping.")
            else:
                actual_files_to_process.append(files[i])
    else:
        actual_files_to_process = files

    for input_file in actual_files_to_process:
        output_file = create_output_filename(
            input_path=input_file, output_dir=output_dir, model_name=model_name
        )
        convert_raw_to_embeddings(
            model_name=model_name,
            model_type=model_type,
            input_path=input_file,
            output_path=output_file,
            device=device,
        )
        print(f"Embeddings for {input_file} created at {output_file}")
    print("Embeddings created successfully")

    return {
        "train": output_file_names[0],
        "dev": output_file_names[1],
        "test": output_file_names[2],
    }


def create_datasets_dataloader(
    task: str,
    model_name: str,
    model_type: str,
    model_max_pos_embeddings: int,
    train_batch: int,
    dev_batch: int,
    test_batch: int,
    train: Tuple[str, str],
    dev: Tuple[str, str],
    test: Tuple[str, str],
    probe_layer: int = -1,
):
    """
    create dataset based on what the task and model
    """

    if task == "pos":
        train_dataset = EmbeddingsDatasetPos(
            h5_py_file=train[0],
            conllu_file=train[1],
            model_name=model_name,
            max_length=model_max_pos_embeddings,
            layer_number=probe_layer,
        )
        dev_dataset = EmbeddingsDatasetPos(
            h5_py_file=dev[0],
            conllu_file=dev[1],
            model_name=model_name,
            max_length=model_max_pos_embeddings,
            layer_number=probe_layer,
        )
        test_dataset = EmbeddingsDatasetPos(
            h5_py_file=test[0],
            conllu_file=test[1],
            model_name=model_name,
            max_length=model_max_pos_embeddings,
            layer_number=probe_layer,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=train_batch, shuffle=True
        )

        dev_dataloader = DataLoader(
            dataset=dev_dataset, batch_size=train_batch, shuffle=True
        )
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=train_batch, shuffle=True
        )
    elif task == "dep_parse":
        train_dataset = DepParseDataset(
            h5_py_file=train[0],
            conllu_file=train[1],
            model_name=model_name,
            max_length=model_max_pos_embeddings,
            layer_number=probe_layer,
        )
        dev_dataset = DepParseDataset(
            h5_py_file=dev[0],
            conllu_file=dev[1],
            model_name=model_name,
            max_length=model_max_pos_embeddings,
            layer_number=probe_layer,
        )
        test_dataset = DepParseDataset(
            h5_py_file=test[0],
            conllu_file=test[1],
            model_name=model_name,
            max_length=model_max_pos_embeddings,
            layer_number=probe_layer,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch, collate_fn=custom_embedding_padding
        )
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=dev_batch, collate_fn=custom_embedding_padding
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=test_batch, collate_fn=custom_embedding_padding
        )
    else:
        raise ValueError("Task not supported")

    return train_dataloader, dev_dataloader, test_dataloader


def init_probe_model_lightning(
    input_dim: int, hidden_dims: List[int], task: str, non_linearity: str = "relu"
):

    if task == "pos":
        from scripts.dataset_script.pos_dataset import POS_TAGS

        output_dim = len(POS_TAGS)
        model = ProbeModelWordLabelLightning(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            non_linearity=non_linearity,
        )
        return model
    elif task == "dep_parse":
        model = ProbeWordPairLabelModelLightning(
            input_dim=input_dim,
            output_dim=int(input_dim / 2),
            hidden_dims=hidden_dims,
            non_linearity=non_linearity,
        )
        return model


def name_of_run(
    model_name: str,
    task: str,
    hiddens: List[int] = [],
    probe_layer: int = -1,
):
    model_name = model_name.replace("/", "_")
    return (
        f"{model_name}_{task}_{'_'.join([str(x) for x in hiddens])}_layer_{probe_layer}"
    )


def run_experiment(
    model_name: str,
    model_type: str,
    task: str,
    train_input: str,
    dev_input: str,
    test_input: str,
    device: str,
    output_dir: str,
    train_batch: int,
    dev_batch: int,
    test_batch: int,
    hiddens: List[int] = [],
    force_create: bool = False,
    probe_layer: int = -1,
):
    """
    Run the experiment
    :param model_name: Name of the model
    :param model_type: Type of the model
    :param model_max_pos_embeddings: Maximum number of embeddings
    :param task: Task to run
    :param train_input: Path to the training file
    :param dev_input: Path to the dev file
    :param test_input: Path to the test file
    :param device: Device to use for embeddings
    :param output_dir: Directory to save the embeddings
    :param train_batch: Batch size for training
    :param dev_batch: Batch size for dev
    :param test_batch: Batch size for test
    :param force_create: Whether to force create the embeddings
    :param hiddens: Hidden dimensions for the probe
    :param probe_layer: Layer to probe
    """

    # Create cache files
    embeddings_files = create_cache_files(
        train_input=train_input,
        dev_input=dev_input,
        test_input=test_input,
        model_name=model_name,
        model_type=model_type,
        device=device,
        output_dir=output_dir,
        force_create=force_create,
    )

    # Initialize model
    model_dict = init_subject_model(model_name=model_name, model_type=model_type)
    text_config = model_dict["config_text"]
    probe_model = init_probe_model_lightning(
        input_dim=text_config.hidden_size,
        hidden_dims=hiddens,
        task=task,
    )

    # Create datasets and dataloaders
    train_dataloader, dev_dataloader, test_dataloader = create_datasets_dataloader(
        task=task,
        model_name=model_name,
        model_type=model_type,
        model_max_pos_embeddings=text_config.max_position_embeddings,
        train_batch=train_batch,
        dev_batch=dev_batch,
        test_batch=test_batch,
        train=(embeddings_files["train"], train_input),
        dev=(embeddings_files["dev"], dev_input),
        test=(embeddings_files["test"], test_input),
        probe_layer=probe_layer,
    )

    # Initialize w&b
    run_name = name_of_run(
        model_name=model_name, task=task, hiddens=hiddens, probe_layer=probe_layer
    )
    wandb_logger = WandbLogger(name=run_name, project="probe_clip_compos_models")

    # Train the model
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    trainer = Trainer(
        logger=wandb_logger, accelerator=device, callbacks=[early_stopping]
    )
    trainer.fit(probe_model, train_dataloader, dev_dataloader)

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, required=True)
    argparser.add_argument("--model_type", type=str, required=True)
    argparser.add_argument("--task", type=str, required=True)
    argparser.add_argument("--train_input", type=str, required=True)
    argparser.add_argument("--dev_input", type=str, required=True)
    argparser.add_argument("--test_input", type=str, required=True)
    argparser.add_argument("--device", type=str, default="cpu")
    argparser.add_argument("--output_dir", type=str, default="data/pos")
    argparser.add_argument("--train_batch", type=int, default=2)
    argparser.add_argument("--dev_batch", type=int, default=2)
    argparser.add_argument("--test_batch", type=int, default=2)
    argparser.add_argument("--force_create", type=bool, default=False)
    argparser.add_argument("--hiddens", nargs="+", type=int, default=[])
    argparser.add_argument("--probe_layer", type=int, default=-1)
    args = argparser.parse_args()
    print(args.hiddens)
    if args.probe_layer == -2:
        # means do all probe layers
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model_name)
        if args.model_type == "clip":
            config = config.text_config

        for l in range(config.num_hidden_layers):
            run_experiment(
                model_name=args.model_name,
                model_type=args.model_type,
                task=args.task,
                train_input=args.train_input,
                dev_input=args.dev_input,
                test_input=args.test_input,
                device=args.device,
                output_dir=args.output_dir,
                train_batch=args.train_batch,
                dev_batch=args.dev_batch,
                test_batch=args.test_batch,
                hiddens=args.hiddens,
                force_create=args.force_create,
                probe_layer=l + 1,
            )
    else:
        run_experiment(
            model_name=args.model_name,
            model_type=args.model_type,
            task=args.task,
            train_input=args.train_input,
            dev_input=args.dev_input,
            test_input=args.test_input,
            device=args.device,
            output_dir=args.output_dir,
            train_batch=args.train_batch,
            dev_batch=args.dev_batch,
            test_batch=args.test_batch,
            hiddens=args.hiddens,
            force_create=args.force_create,
            probe_layer=args.probe_layer,
        )
