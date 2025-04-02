import torch
import numpy as np
from typing import *

from datasets import load_dataset

from scripts.ad_hoc_gaus_classifier import *

from scripts.utils.model_init import init_model_sd_vae

from torch import Tensor

import pandas as pd
import numpy as np

from skimage.metrics import structural_similarity as ssim


def calculate_ssim(img1, img2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: First image as a 2D or 3D NumPy array (H x W x C for RGB).
        img2: Second image as a 2D or 3D NumPy array (H x W x C for RGB).

    Returns:
        ssim: SSIM value between img1 and img2.
    """
    # Ensure images are the same size
    assert img1.shape == img2.shape, "Input images must have the same dimensions"

    # Constants to avoid division by zero
    C1 = 0.01**2
    C2 = 0.03**2

    # def ssim_channel(channel1, channel2):
    #     """Calculate SSIM for a single channel."""
    #     mu1 = np.mean(channel1)
    #     mu2 = np.mean(channel2)
    #     sigma1_sq = np.var(channel1)
    #     sigma2_sq = np.var(channel2)
    #     sigma12 = np.cov(channel1.flatten(), channel2.flatten())[0, 1]

    #     numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    #     denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    #     return numerator / denominator

    # If grayscale (2D array), compute SSIM directly
    if len(img1.shape) == 2:
        return ssim(img1, img2)

    # For multi-channel (3D array), compute SSIM for each channel and average
    channels = img1.shape[2]

    ssim_values = [
        ssim(
            img1[..., i],
            img2[..., i],
            data_range=max(img1[..., i].max(), img2[..., i].max())
            - min(img1[..., i].min(), img2[..., i].min()),
        )
        for i in range(channels)
    ]
    return np.mean(ssim_values)


def make_dataset_splits(dataset_name):
    # load the shuffled dataset and break into splits
    dataset = load_dataset(
        dataset_name, cache_dir="/scratch/shashwat.s/.cache/huggingface/"
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset["train"]

    train_test_split = dataset.train_test_split(test_size=0.2, shuffle=False)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    val_test_split = test_dataset.train_test_split(test_size=0.5, shuffle=False)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    return train_dataset, val_dataset, test_dataset


def conduct_interpolations(inp1, inp2, num_interpolations, device) -> List[Tensor]:
    # Conduct linear interpolations between inp1 and inp2
    interpolations = []
    for alpha in np.linspace(0, 1, num_interpolations):
        alpha = torch.tensor(alpha).to(device)
        interpolation = alpha * inp1 + (1 - alpha) * inp2
        interpolations.append(interpolation)
    return interpolations


def get_outputs_from_vae_model(
    pipe, inp1, inp2, num_interpolations, device
) -> List[Tensor]:
    # Get the outputs of the VAE model for the interpolations
    vae_model = pipe.vae
    interpolations = conduct_interpolations(inp1, inp2, num_interpolations, device)
    outputs = []
    for interpolation in interpolations:
        output = pipe.image_processor.postprocess(
            vae_model.decode(interpolation).sample
        )
        outputs.append(output)
    return outputs


def evaluate_linearity(constructed_out, out):
    ssim_total = 0
    for i in range(len(constructed_out)):
        ssim_total += calculate_ssim(constructed_out[i], out[i])
    ssim_total /= len(constructed_out)

    return {
        "mse": ((constructed_out - out) ** 2).mean(),
        "ssim_total": ssim_total,
    }


from tqdm import tqdm


def main(num_samples: int = 1000):
    train, test, val = make_dataset_splits("microsoft/cats_vs_dogs")
    train_path = "/scratch/shashwat.s/embed_cache/vae_train_dataset_cache.h5"
    val_path = "/scratch/shashwat.s/embed_cache/vae_val_dataset_cache.h5"
    test_path = "/scratch/shashwat.s/embed_cache/vae_test_dataset_cache.h5"

    train_X, train_y = process_for_wasserstein_space(train_path, train)

    train_X_dog = train_X[:, : train_X.shape[1] // 2][train_y == 0]
    train_X_cat = train_X[:, train_X.shape[1] // 2 :][train_y == 1]

    # samples a bunch of indice
    np.random.seed(42)
    dog_indices = np.random.choice(train_X_dog.shape[0], num_samples)
    cat_indices = np.random.choice(train_X_cat.shape[0], num_samples)

    vae, feature_extractor, pipe = init_model_sd_vae()
    vae = vae.to("cuda")

    results = []

    for i in tqdm(range(num_samples)):
        inp1 = (
            train_X_dog[dog_indices[i]]
            .view(1, 4, 28, 28)
            .to(dtype=torch.float16)
            .to(vae.device)
        )
        inp2 = (
            train_X_cat[cat_indices[i]]
            .view(1, 4, 28, 28)
            .to(dtype=torch.float16)
            .to(vae.device)
        )

        out1 = torch.tensor(
            np.array(pipe.image_processor.postprocess(vae.decode(inp1).sample)[0]),
            device=vae.device,
        )
        out2 = torch.tensor(
            np.array(pipe.image_processor.postprocess(vae.decode(inp2).sample)[0]),
            device=vae.device,
        )
        # print(type(out1), out1, out1[0])

        constructed_out = get_outputs_from_vae_model(pipe, inp1, inp2, 10, "cuda")
        out = conduct_interpolations(out1, out2, 10, "cuda")
        # out = torch.stack(out).squeeze()

        # Test the linearity of the model

        out = np.stack([np.array(oo.cpu()) for oo in out], axis=0)

        constructed_out = np.stack(
            [np.array(consout[0]) for consout in constructed_out], axis=0
        )

        # constructed_out = torch.stack(constructed_out).squeeze()
        eval_metrics = evaluate_linearity(constructed_out, out)

        results.append(eval_metrics)

    # make a df with the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results_linearity_{num_samples}.csv")
    return results_df


import argparse

if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("--num_samples", type=int, default=100)
    args = parse.parse_args()

    with torch.no_grad():
        df = main(num_samples=args.num_samples)

    mses = torch.tensor(df["mse"])

    # print(f"Mean MSE: {mses.mean()}")
    # print(f"Std MSE: {mses.std()}")

    print("Mean")
    print(df.mean(axis=0))

    print("Standard division")
    print(df.std(axis=0))
