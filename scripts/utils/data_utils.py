from conllu import TokenList
import torch
from typing import *


def deal_with_double_punctuation_case(tokenlist: TokenList, task: str):
    """
    This function deals with the case where there are two punctuation marks in a row.
    For example '."'
    this trips up tokenizer because the UD structure seperates them.
    Simple fix is that we can just return a new Token list with consecutive puntuations merged
    """
    new_token_list = []
    last_is_punct = False

    # this is because indexing is changing and we have to fix that for the dependency parsing stuff
    index_map = []
    # print("hi")

    for i, token in enumerate(tokenlist):
        if last_is_punct:
            if token["upos"] == "PUNCT":
                new_token_list[-1]["form"] = new_token_list[-1]["form"] + token["form"]
                new_token_list[-1]["lemma"] = (
                    new_token_list[-1]["lemma"] + token["lemma"]
                )
                new_token_list[-1]["misc"] = token["misc"]
                index_map.append(index_map[-1] - 1)
            else:
                new_token_list.append(token)
                index_map.append(index_map[-1])
        else:
            new_token_list.append(token)
            if len(index_map) != 0:
                index_map.append(index_map[-1])
            else:
                index_map.append(0)
        last_is_punct = token["upos"] == "PUNCT"
    # print(index_map)
    # remap the dependency parses
    if task == "dep_parse":
        assert len(tokenlist) == len(index_map)
        for i, token in enumerate(new_token_list):
            if token["head"] is not None:
                assert token["head"] < len(tokenlist)
                token["head"] = index_map[token["head"]] + token["head"]

                token["deps"] = [(token["deprel"], token["head"])]
    return new_token_list


from collections import defaultdict


def map_to_subword_generic(batch_encode_obj):
    mapping = defaultdict(list)
    word_maps = batch_encode_obj.word_ids()
    # print(word_maps)
    for i, maps in enumerate(word_maps):
        if maps is not None:
            mapping[maps].append(i)
    return mapping


def pool_for_words(
    token_obj,
    embeddings: torch.Tensor,
    pool_type: str = "mean",
) -> torch.Tensor:

    word_maps = map_to_subword_generic(token_obj)
    pooled = []
    for word_map in word_maps:
        word_indices = word_maps[word_map]
        word_embedding = embeddings[word_indices]
        if pool_type == "mean":
            word_embedding = word_embedding.mean(dim=0)
        elif pool_type == "max":
            word_embedding = word_embedding.max(dim=0).values
        pooled.append(word_embedding)
    return torch.stack(pooled)


from torch.nn.utils.rnn import pad_sequence


def custom_embedding_padding(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    padding_value_x: float = 0.0,
    padding_value_y: int = -1,
):
    """
    Inputs:
    Tuple: (embedding_list: List of torch tensors of shape (seq_len, d)
    y_list: List of torch tensors of shape (seq_len,))
    padding_value_x: float
    padding_value_y: int
    Outputs:
    padded_x: torch tensor of shape (batch_size, max_seq_len, d)
    padded_y: torch tensor of shape (batch_size, max_seq_len)
    """

    embedding_list = [x[0] for x in batch]
    y_list = [x[1] for x in batch]
    padded_x = pad_sequence(
        embedding_list, batch_first=True, padding_value=padding_value_x
    )

    padded_y = pad_sequence(y_list, batch_first=True, padding_value=padding_value_y)

    return padded_x, padded_y
