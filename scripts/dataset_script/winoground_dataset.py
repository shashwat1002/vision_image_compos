from torch.utils.data import Dataset
import h5py
import torch


class WinogroundEmbeddingDataset(Dataset):
    def __init__(self, text_embed_path, image_embed_path, layer_number=-1):
        self.text_embed_path = text_embed_path
        self.image_embed_path = image_embed_path

        text_hf = h5py.File(text_embed_path, "r")
        image_hf = h5py.File(image_embed_path, "r")

        single_layer_text_list = []
        image_feature_list = []

        indices_text = list(text_hf.keys())
        indices_text = [int(i) for i in indices_text]

        for i in indices_text:
            single_layer_text_list.append(text_hf.get(str(i))[:, layer_number, :])

        indices_image = list(image_hf.keys())
        indices_image = [int(i) for i in indices_image]

        for i in indices_image:
            image_feature_list.append(image_hf.get(str(i))[:, :])

        text_hf.close()
        image_hf.close()

        self.single_layer_text_list = single_layer_text_list
        self.image_feature_list = image_feature_list

    def __len__(self):
        return len(self.single_layer_text_list)

    def __getitem__(self, index):
        text_rep = torch.tensor(self.single_layer_text_list[index])
        image_rep = torch.tensor(self.image_feature_list[index])

        # concatenate them on dim=0 -> c1, c2, i1, i2
        return torch.cat([text_rep, image_rep], dim=0)
