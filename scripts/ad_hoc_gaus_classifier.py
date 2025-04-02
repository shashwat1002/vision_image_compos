from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import torch
from typing import List, Tuple
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm


def process_gaussian_train_file(
    train_file: str,
    dataset_obj: Dataset,
) -> Tuple[DiagonalGaussianDistribution, DiagonalGaussianDistribution]:
    """
    Make an average gaussian for both classes
    Inputs:
    * train_file: str: path to the training file
    Outputs:
    (DiagonalGaussianDistribution, DiagonalGaussianDistribution): the Gaussian distribution for the training file
    """
    hf = h5py.File(train_file, "r")
    file_indices = list(hf.keys())
    file_indices = [int(i) for i in file_indices]
    file_indices.sort()

    class_indices = []
    embeddings = []

    c1_mean, c2_mean = None, None
    c1_var, c2_var = None, None

    for i in tqdm(file_indices):
        class_label = dataset_obj[i]["labels"]
        class_indices.append(class_label)
        params_gaussian = torch.tensor(hf.get(str(i))[:, :, :, :])
        gaus_obj = DiagonalGaussianDistribution(params_gaussian)

        if c1_mean is None:
            c1_mean = torch.zeros_like(gaus_obj.mean)
            c1_var = torch.zeros_like(gaus_obj.var)
            c2_mean = torch.zeros_like(gaus_obj.mean)
            c2_var = torch.zeros_like(gaus_obj.var)

        if class_label == 0:
            c1_mean += gaus_obj.mean
            c1_var += gaus_obj.var
        else:
            c2_mean += gaus_obj.mean
            c2_var += gaus_obj.var

    hf.close()

    c1_mean /= len(class_indices)
    c1_var /= len(class_indices) ** 2
    c2_mean /= len(class_indices)
    c2_var /= len(class_indices) ** 2

    c1_var = torch.log(c1_var)
    c2_var = torch.log(c2_var)

    # turn into a gaussian object
    c1_gaus = DiagonalGaussianDistribution(torch.cat([c1_mean, c1_var], dim=1))
    c2_gaus = DiagonalGaussianDistribution(torch.cat([c2_mean, c2_var], dim=1))

    return c1_gaus, c2_gaus


def process_gaussian_mixture_approximation(
    train_file: str, dataset_obj: Dataset
) -> Tuple[DiagonalGaussianDistribution, DiagonalGaussianDistribution]:

    hf = h5py.File(train_file, "r")
    file_indices = list(hf.keys())
    file_indices = [int(i) for i in file_indices]
    file_indices.sort()

    class_indices = []
    mean_embeds = []
    var_embeds = []

    for i in tqdm(file_indices):
        class_label = dataset_obj[i]["labels"]
        class_indices.append(class_label)
        params_gaussian = torch.tensor(hf.get(str(i))[:, :, :, :])
        gaus_obj = DiagonalGaussianDistribution(params_gaussian)

        mean_embeds.append(gaus_obj.mean)
        var_embeds.append(gaus_obj.var)
    
    c1_mean = torch.stack(mean_embeds).mean(dim=0)
    c1_var = torch.stack(var_embeds).mean(dim=0)
    c1_var_of_mean = mean_embeds.cov(dim=0)
    

    hf.close()

    c1_mean /= len(class_indices)
    c1_var /= len(class_indices) ** 2
    c2_mean /= len(class_indices)
    c2_var /= len(class_indices) ** 2

    c1_var = torch.log(c1_var)
    c2_var = torch.log(c2_var)

    # turn into a gaussian object
    c1_gaus = DiagonalGaussianDistribution(torch.cat([c1_mean, c1_var], dim=1))
    c2_gaus = DiagonalGaussianDistribution(torch.cat([c2_mean, c2_var], dim=1))

    return c1_gaus, c2_gaus


def process_for_wasserstein_space(
    train_file: str, dataset_obj: Dataset
) -> Tuple[torch.Tensor, torch.Tensor]:

    hf = h5py.File(train_file, "r")
    file_indices = list(hf.keys())
    file_indices = [int(i) for i in file_indices]
    file_indices.sort()

    class_indices = []
    embeddings = []

    for i in tqdm(file_indices):
        class_label = dataset_obj[i]["labels"]
        class_indices.append(class_label)
        params_gaussian = torch.tensor(hf.get(str(i))[:, :, :, :])
        gaus_obj = DiagonalGaussianDistribution(params_gaussian)

        params_gaussian = torch.cat(
            [
                gaus_obj.mean.flatten(),
                gaus_obj.std.flatten(),
            ],
            dim=-1,
        )

        embeddings.append(params_gaussian)

    hf.close()

    return torch.stack(embeddings), torch.tensor(class_indices)


class TrainedGausClassifier:
    def __init__(self):
        self.gaus_objs = []

    def fit(self, train_file: str, dataset_obj: Dataset):
        c1_gaus, c2_gaus = process_gaussian_train_file(train_file, dataset_obj)
        self.gaus_objs.append(c1_gaus)
        self.gaus_objs.append(c2_gaus)

    def predict(self, test_file: str, dataset_obj: Dataset) -> List[int]:
        hf = h5py.File(test_file, "r")
        file_indices = list(hf.keys())
        file_indices = [int(i) for i in file_indices]
        file_indices.sort()

        class_indices = []

        for i in tqdm(file_indices):
            class_label = dataset_obj[i]["labels"]
            params_gaussian = torch.tensor(hf.get(str(i))[:, :, :, :])
            other_obj = DiagonalGaussianDistribution(params_gaussian)

            # compute the distance to the two gaussians
            distances = []
            for gaus_obj in self.gaus_objs:
                distances.append(gaus_obj.kl(other_obj))

            # get the class with the minimum distance
            class_indices.append(distances.index(min(distances)))

        hf.close()

        return class_indices

    def predict_single(self, other_obj: DiagonalGaussianDistribution) -> int:
        distances = []
        for gaus_obj in self.gaus_objs:
            distances.append(gaus_obj.kl(other_obj))
        return distances.index(min(distances))
