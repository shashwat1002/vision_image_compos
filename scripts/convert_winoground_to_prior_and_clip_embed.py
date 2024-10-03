from .utils.model_init import init_subject_model, init_diffusion_prior_model
import h5py
import torch
from tqdm import tqdm
from .utils.feature_extraction import (
    get_openai_clip_image_embeds,
    sample_prior_embeddings,
)
import clip


def create_output_filename(
    input_path: str,
    output_dir: str,
    proj: bool = True,
    model_path: str = None,
    predicted: bool = True,
    split: float = 1.0,
    agg: str = "default",
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
    ckpt = model_path.split("/")[-1].split(".")[0]
    return (
        f"{output_dir}/{ckpt}_{basename}_{predicted}_{agg}_proj_{proj}_{split}_text.h5",
        f"{output_dir}/{ckpt}_{basename}_{split}_img.h5",
    )


def convert_raw_to_embeddings(
    dataset_split,
    model_path: str,
    output_paths: tuple[str, str],  # image file and text file
    device: str = "cpu",
    predicted: bool = True,  # diffusion predicted embeds
    agg: str = "default",  # aggregation method
    num_samples: int = 2,
):

    diffusion_prior, clip_model = init_diffusion_prior_model(
        path=model_path, device=device
    )

    _, preprocess = clip.load("ViT-L/14")
    del _

    text_embed_path, image_embed_path = output_paths

    with h5py.File(text_embed_path, "w") as fout:
        with h5py.File(image_embed_path, "w") as fimg:
            for index, example in tqdm(enumerate(dataset_split)):

                # unpack the sample
                c1 = example["caption_0"]
                c2 = example["caption_1"]
                i1 = example["image_0"].convert("RGB")
                i2 = example["image_1"].convert("RGB")

                # get the image embedding
                i1_embed = get_openai_clip_image_embeds(
                    clip_model, preprocess, i1, device=device
                )
                i2_embed = get_openai_clip_image_embeds(
                    clip_model, preprocess, i2, device=device
                )

                c1_tokenized = clip.tokenize(c1).to(device)
                c2_tokenized = clip.tokenize(c2).to(device)

                # get the text embedding
                if predicted:
                    if agg == "default":
                        c1_embed = diffusion_prior.sample(
                            c1_tokenized, num_samples_per_batch=num_samples
                        )
                        c2_embed = diffusion_prior.sample(
                            c2_tokenized, num_samples_per_batch=num_samples
                        )
                        # c_tensor = diffusion_prior.sample(
                        #     torch.stack([c1_tokenized, c2_tokenized]),
                        #     num_samples_per_batch=num_samples,
                        # )
                    elif agg == "no_agg":
                        c1_embeds_all = sample_prior_embeddings(
                            prior_model=diffusion_prior,
                            text=c1_tokenized,
                            num_samples_per_batch=num_samples,
                            device=device,
                        )
                        c1_embed = c1_embeds_all
                        c2_embeds_all = sample_prior_embeddings(
                            prior_model=diffusion_prior,
                            text=c2_tokenized,
                            num_samples_per_batch=num_samples,
                            device=device,
                        )
                        c2_embed = c2_embeds_all
                    else:
                        c1_embeds_all = sample_prior_embeddings(
                            prior_model=diffusion_prior,
                            text=c1_tokenized,
                            num_samples_per_batch=num_samples,
                            device=device,
                        )
                        c1_embed = c1_embeds_all.mean(dim=0, keepdim=True)
                        c2_embeds_all = sample_prior_embeddings(
                            prior_model=diffusion_prior,
                            text=c2_tokenized,
                            num_samples_per_batch=num_samples,
                            device=device,
                        )
                        c2_embed = c2_embeds_all.mean(dim=0, keepdim=True)
                        print(c1_embed.shape, c2_embed.shape)
                else:
                    c1_embed = clip_model.clip.encode_text(c1_tokenized)
                    c2_embed = clip_model.clip.encode_text(c2_tokenized)

                # make a c tensor with the two captions
                c_tensor = torch.stack([c1_embed, c2_embed])

                # make an i tensor with the two images
                i_tensor = torch.stack([i1_embed, i2_embed])

                dset = fout.create_dataset(
                    str(index), (2, c_tensor.shape[-2], c_tensor.shape[-1])
                )

                dset[:, :, :] = c_tensor.unsqueeze(dim=0).cpu().numpy()

                dimg = fimg.create_dataset(str(index), (2, i_tensor.shape[-1]))
                dimg[:, :] = i_tensor.squeeze().cpu().numpy()
