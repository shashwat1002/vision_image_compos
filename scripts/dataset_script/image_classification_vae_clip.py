from torch.utils.data import Dataset
from datasets import load_dataset
import h5py
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import torch


class VAEEmbed(Dataset):
    def __init__(self, dataset, file_name, only_mean=True, pooling="concat"):
        """
        Inputs:
        * dataset_name: str: name of the dataset
        * file_name: str: path to the file
        * only_mean: bool: whether to use only the mean of the Gaussian distribution
        * pooling: str: type of pooling to use (mean, concat)
        """

        hf = h5py.File(file_name, "r")
        file_indices = list(hf.keys())
        file_indices = [int(i) for i in file_indices]
        file_indices.sort()

        self.class_indices = []
        self.embeddings = []
        self.pooling = pooling

        self.dim = 0

        for i in file_indices:
            class_label = dataset[i]["labels"]
            self.class_indices.append(class_label)
            params_gaussian = torch.tensor(hf.get(str(i))[:, :])
            # print(params_gaussian.shape)
            gaus_obj = DiagonalGaussianDistribution(params_gaussian)
            if only_mean:
                params_gaussian = 1 / (1 + torch.exp(-gaus_obj.mean / gaus_obj.var))

                # print(params_gaussian.shape)
            else:
                # trust me this is derived from the wasserstein metric
                params_gaussian = torch.cat(
                    [
                        gaus_obj.mean.flatten(),
                        gaus_obj.var.flatten(),
                    ],
                    dim=-1,
                )
                print("hi", params_gaussian.shape)
            # params_gaussian = params_gaussian.view(params_gaussian.shape[0], -1)
            if pooling == "mean" and not only_mean:
                params_gaussian = torch.mean(params_gaussian, dim=0)
            elif pooling == "concat":
                params_gaussian = torch.flatten(params_gaussian)
            print(params_gaussian.shape)
            self.embeddings.append(params_gaussian)
            self.dim = params_gaussian.shape[-1]

        self.num_classes = len(set(self.class_indices))
        hf.close()

    def __len__(self):
        return len(self.class_indices)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.class_indices[idx]


class EncEmberDataset(Dataset):
    def __init__(self, dataset, file_name):
        """
        Inputs:
        * dataset_name: str: name of the dataset
        * file_name: str: path to the file
        """

        hf = h5py.File(file_name, "r")
        file_indices = list(hf.keys())
        file_indices = [int(i) for i in file_indices]
        file_indices.sort()

        self.class_indices = []
        self.embeddings = []
        self.dim = 0

        for i in file_indices:
            class_label = dataset[i]["labels"]
            self.class_indices.append(class_label)
            em = hf.get(str(i))[:]
            self.embeddings.append(em)
            self.dim = em.shape[-1]

        hf.close()

    def __len__(self):
        return len(self.class_indices)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), self.class_indices[idx]
