import torch
from torch.utils.data import Dataset
from conllu import parse_incr
import h5py
from collections import defaultdict
from transformers import AutoTokenizer
import numpy as np

from ..utils.data_utils import (
    deal_with_double_punctuation_case,
    deal_with_double_punctuation_case,
    pool_for_words,
    map_to_subword_generic,
)


class DepParseDataset(Dataset):
    def __init__(
        self, h5_py_file, conllu_file, model_name, max_length, layer_number: int = -1
    ):
        self.h5_py_file = h5_py_file
        self.conllu_file = conllu_file
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # store the parse trees with respect to entries
        self.entries = []  # each element is conllu tokenlist
        with open(conllu_file, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                self.entries.append(tokenlist)

        hf = h5py.File(h5_py_file, "r")
        indices = list(hf.keys())
        single_layer_features_list = []

        indices = [int(i) for i in indices]
        indices.sort()

        final_processed_embeds = []

        self.dataset = []

        num_mismatch = 0

        # store embedding corresponding to each conllu entry
        for i in indices:
            single_layer_features_list.append(hf.get(str(i))[layer_number])

        samples_abandoned = 0
        # iterate over the list of tokenlists
        for i, tokenlist in enumerate(self.entries):
            # print(i)
            text_tokenlist = tokenlist.metadata["text"]  # get the text

            # getting the tokenizer object corresponding to the subword tokenizer
            token_obj = self.tokenizer(
                text_tokenlist,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            # get all the embeddings for a sentence
            embeddings = torch.tensor(single_layer_features_list[i])
            assert embeddings.shape[0] == len(token_obj["input_ids"][0])

            # pool the embeddings for each word

            aligned_embeddings = pool_for_words(token_obj, embeddings, pool_type="mean")

            # store the pooled embeddings
            final_processed_embeds.append(aligned_embeddings)

            abandon_flag = False
            for token in tokenlist:
                print(token, token["head"])
                if token["head"] is not None and token["head"] >= len(tokenlist):
                    abandon_flag = True
                    break

            if abandon_flag:
                samples_abandoned += 1
                continue
            # align embedding to pos tags
            tokenlist = deal_with_double_punctuation_case(tokenlist, task="dep_parse")

            heads = [
                token["head"] if token["head"] is not None else -1
                for token in tokenlist
            ]
            print(heads)
            heads = torch.tensor(heads)
            deprel = [token["deprel"] for token in tokenlist]  # not used yet

            if aligned_embeddings.shape[0] != len(heads):
                print("Mismatch", aligned_embeddings.shape, len(heads))
                print(text_tokenlist)
                num_mismatch += 1
                continue

            self.dataset.append(
                (
                    aligned_embeddings,
                    heads,
                )
            )
        print("Mismatch", num_mismatch)
        print("Abandons", samples_abandoned)
        hf.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
