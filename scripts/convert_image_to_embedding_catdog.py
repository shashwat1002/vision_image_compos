from .utils.model_init import init_model_sd_vae
from .utils.feature_extraction import get_latent_distribution
import h5py
import torch


def create_output_filename(dataset_name:str, split:str, output_dir: str, model_name: str) -> str:
    """
    Create the output filename for the embeddings
    Inputs:
    * input_path: str: path to the input file
    * output_dir: str: path to the output directory
    * model_name: str: name of the model
    Outputs:
    str: the output filename
    """
    model_name = model_name.replace("/", "_")
    return f"{output_dir}/{model_name}_{dataset_name}_{split}.h5"


from tqdm import tqdm


def convert_raw_to_embeddings(
    dataset_split,
    output_path: str,
    device: str = "cpu",
):

    model_vae, feature_extractor = init_model_sd_vae()  # use default params

    with torch.no_grad():
        with h5py.File(output_path, "w") as fimg:
            for index, example in tqdm(enumerate(dataset_split)):

                i = example["image"]
                i = feature_extractor(images=i, return_tensors="pt").pixel_values
                i = i.to(device, dtype=torch.float16)

                encoder_distr = model_vae.encode(i).latent_dist.parameters

                dset = fimg.create_dataset(
                    str(index),
                    encoder_distr.shape,
                )

                dset[:, :, :, :] = encoder_distr.cpu().numpy()


from scripts.utils.model_init import init_subject_model


def convert_raw_to_embeddings_image_encs(
    model_name,
    model_type,
    dataset_split,
    split,
    output_path: str,
    device: str = "cpu",
):

    model_dict = init_subject_model(model_name, model_type, device=device)
    print(model_dict)
    with torch.no_grad():
        with h5py.File(output_path, "w") as fimg:
            for index, example in tqdm(enumerate(dataset_split)):
                i = example["image"]
                i = model_dict["processor"](images=i, return_tensors="pt")
                # print(i)
                # i = i.pooler_output
                i = i.to(device)
                embed = model_dict["model_image"](**i).pooler_output
                dset = fimg.create_dataset(
                    str(index),
                    embed.shape,
                )

                dset[:] = embed.cpu().numpy()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
    )

    parser.add_argument(
        "--split",
        type=str,
        help="The dataset split to use",
        default="",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to use",
        default="microsoft/cats_vs_dogs",
    )

    args = parser.parse_args()

    output_path = create_output_filename(
        input_path=args.input_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

    from datasets import load_dataset

    dataset_split = load_dataset(args.dataset, split=args.split)

    convert_raw_to_embeddings(
        dataset_split=dataset_split,
        output_path=output_path,
    )
