from conllu import TokenList
import torch
from typing import *


def deal_with_double_punctuation_case(tokenlist: TokenList):
    """
    This function deals with the case where there are two punctuation marks in a row.
    For example '."'
    this trips up tokenizer because the UD structure seperates them.
    Simple fix is that we can just return a new Token list with consecutive puntuations merged
    """
    new_token_list = []
    last_is_punct = False
    for i, token in enumerate(tokenlist):
        if last_is_punct:
            if token["upos"] == "PUNCT":
                new_token_list[-1]["form"] = new_token_list[-1]["form"] + token["form"]
                new_token_list[-1]["lemma"] = (
                    new_token_list[-1]["lemma"] + token["lemma"]
                )
                new_token_list[-1]["misc"] = token["misc"]
            else:
                new_token_list.append(token)
        else:
            new_token_list.append(token)
        last_is_punct = token["upos"] == "PUNCT"
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
