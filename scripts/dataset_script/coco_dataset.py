from torch.utils.data import Dataset
import h5py
import torch


class CoCoEmbeddingDataset(Dataset):
    def __init__(self, text_embed_path, image_embed_path):
        self.text_embed_path = text_embed_path
        self.image_embed_path = image_embed_path

        text_hf = h5py.File(text_embed_path, "r")
        image_hf = h5py.File(image_embed_path, "r")

        single_layer_text_list = []
        image_feature_list = []

        indices_text = list(text_hf.keys())
        indices_text = [int(i) for i in indices_text]
        # sort the indices
        indices_text.sort()

        for i in indices_text:
            single_layer_text_list.append(
                text_hf.get(str(i))[:, :]
            )  # sequence length, dimensions

        indices_image = list(image_hf.keys())
        indices_image = [int(i) for i in indices_image]
        # sort the indices
        indices_image.sort()

        for i in indices_image:
            image_feature_list.append(image_hf.get(str(i))[:])  # dimensions

        text_hf.close()
        image_hf.close()

        self.single_layer_text_list = single_layer_text_list
        self.image_feature_list = image_feature_list

    def __len__(self):
        return len(self.single_layer_text_list)

    def __getitem__(self, index):
        text_rep = torch.tensor(self.single_layer_text_list[index])
        image_rep = torch.tensor(self.image_feature_list[index])

        return {
            "input_embed": text_rep.unsqueeze(0),
            "attention_mask": torch.ones(text_rep.shape[-2]).unsqueeze(-1),
            "image_embed": image_rep.unsqueeze(0),
        }


# padding function
def custom_collate_fn(batch):
    x = [x["input_embed"].squeeze() for x in batch]
    y = [x["image_embed"] for x in batch]
    mask = [x["attention_mask"] for x in batch]
    print(len(x), len(y), len(mask))
    print(x[0].shape, x[1].shape, y[0].shape, mask[0].shape)
    input_embed = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=0.0
    ).squeeze()
    attention_mask = (
        1
        - torch.nn.utils.rnn.pad_sequence(
            mask, batch_first=True, padding_value=0
        ).squeeze()
    )
    to_ret = {
        "input_embed": input_embed,
        "image_embed": torch.stack(y).squeeze(),
        "attention_mask": attention_mask,
    }
    print("RET")
    print(
        to_ret["input_embed"].shape,
        to_ret["image_embed"].shape,
        to_ret["attention_mask"].shape,
    )
    return to_ret
