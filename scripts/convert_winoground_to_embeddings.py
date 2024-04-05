from .utils.model_init import init_subject_model
import h5py
import torch
from tqdm import tqdm
from .utils.feature_extraction import get_embedding_wino_eval


def create_output_filename(
    input_path: str, output_dir: str, model_name: str, pool_strat: str = "pooler"
) -> str:
    """
    Create the output filename for the embeddings
    Inputs:
    * input_path: str: path to the input file
    * output_dir: str: path to the output directory
    * model_name: str: name of the model
    Outputs:
    (str, str): the output filename for text and image
    """
    basename = input_path.split("/")[-1].split(".")[0]
    model_name = model_name.replace("/", "_")
    return (
        f"{output_dir}/{model_name}_{basename}_{pool_strat}_text.h5",
        f"{output_dir}/{model_name}_{basename}_img.h5",
    )


def convert_raw_to_embeddings(
    dataset_split,
    model_name: str,
    model_type: str,
    output_paths: tuple[str, str],  # image file and text file
    device: str = "cpu",
):

    model_init_dict = init_subject_model(
        model_name=model_name, model_type=model_type, device=device
    )

    text_config = model_init_dict["config_text"]

    num_layers = text_config.num_hidden_layers + 1  # +1 for the embeddings
    feature_count = text_config.hidden_size

    text_embed_path, image_embed_path = output_paths

    with h5py.File(text_embed_path, "w") as fout:
        with h5py.File(image_embed_path, "w") as fimg:
            for index, example in tqdm(enumerate(dataset_split)):

                # unpack the sample
                c1 = example["caption_0"]
                c2 = example["caption_1"]
                i1 = example["image_0"].convert("RGB")
                i2 = example["image_1"].convert("RGB")

                # get the embeddings:
                c1_embed_list, c2_embed_list, i1_embed, i2_embed = (
                    get_embedding_wino_eval(model_init_dict, c1, c2, i1, i2)
                )

                c1_embed_tensor = torch.stack(
                    c1_embed_list
                )  # (num_layers, feature_count)
                c2_embed_tensor = torch.stack(
                    c2_embed_list
                )  # (num_layers, feature_count)

                # make a c tensor with the two captions
                c_tensor = torch.stack([c1_embed_tensor, c2_embed_tensor])

                # make an i tensor with the two images
                i_tensor = torch.stack([i1_embed, i2_embed])

                dset = fout.create_dataset(str(index), (2, num_layers, feature_count))

                dset[:, :, :] = c_tensor.squeeze().cpu().numpy()

                dimg = fimg.create_dataset(str(index), (2, feature_count))
                dimg[:, :] = i_tensor.squeeze().cpu().numpy()
