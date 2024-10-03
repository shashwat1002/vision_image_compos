from .utils.model_init import init_subject_model
import h5py
import torch
from tqdm import tqdm
from .utils.feature_extraction import get_embedding_wino_eval


def create_output_filename(
    input_path: str,
    output_dir: str,
    text_model_name: str,
    image_model_name: str,
    split_name: str,
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
    text_model_name = text_model_name.replace("/", "_")
    image_model_name = image_model_name.replace("/", "_")

    return (
        f"{output_dir}/{text_model_name}_{basename}_text_{split_name}.h5",
        f"{output_dir}/{image_model_name}_{basename}_img{split_name}.h5",
    )


def convert_raw_to_embeddings(
    dataset_split,
    text_model_name: str,
    image_model_name: str,
    output_paths: tuple[str, str],  # image file and text file
    device: str = "cpu",
):

    text_model_init_dict = init_subject_model(
        model_name=text_model_name, model_type="text", device=device
    )

    image_model_init_dict = init_subject_model(
        model_name=image_model_name, model_type="image", device=device
    )

    text_config = text_model_init_dict["config_text"]
    text_tokenizer = text_model_init_dict["tokenizer"]
    text_model = text_model_init_dict["model_text"]

    image_config = image_model_init_dict["config_image"]
    image_processor = image_model_init_dict["processor"]
    image_model = image_model_init_dict["model_image"]

    feature_count_text = text_config.hidden_size
    feature_count_image = image_config.hidden_size

    text_embed_path, image_embed_path = output_paths

    with h5py.File(text_embed_path, "w") as fout:
        with h5py.File(image_embed_path, "w") as fimg:
            for index, example in tqdm(enumerate(dataset_split)):

                # unpack the sample
                c = example["sentences"]["raw"]
                i = example["image"].convert("RGB")

                # get the text embeddings
                c_tokenize = text_tokenizer([c], return_tensors="pt").to(device=device)
                sequence_length = c_tokenize["input_ids"].shape[-1]
                with torch.no_grad():
                    c_embed = (
                        text_model(**c_tokenize)
                        .last_hidden_state.squeeze()
                        .cpu()
                        .detach()
                        .numpy()
                    )

                # get the image embeddings
                i_processed = image_processor(images=i, return_tensors="pt").to(
                    device=device
                )
                with torch.no_grad():
                    i_embed = (
                        image_model(**i_processed)
                        .last_hidden_state.squeeze()[0, :]
                        .cpu()
                        .detach()
                        .numpy()
                    )

                dset = fout.create_dataset(
                    str(index), (sequence_length, feature_count_text)
                )

                dset[:, :] = c_embed

                dimg = fimg.create_dataset(str(index), (i_embed.shape[-1]))
                dimg[:] = i_embed
