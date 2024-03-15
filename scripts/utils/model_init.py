from typing import *
from torch.nn import Module
from torch import nn


def probe_block(input_dim: int, output_dim: int, non_linearity: str) -> Module:

    non_linearity_layer = None
    if non_linearity == "relu":
        non_linearity_layer = nn.ReLU()
    elif non_linearity == "tanh":
        non_linearity_layer = nn.Tanh()
    elif non_linearity == "gelu":
        non_linearity_layer = nn.GELU()

    return nn.Sequential(nn.Linear(input_dim, output_dim), non_linearity_layer)


def probe_model(
    input_dim: int, output_dim: int, hidden_dims: List[int], non_linearity: str
) -> Module:
    layers = []
    last_hidden = input_dim
    # current = None
    for hidden_dim in hidden_dims:
        layers.append(probe_block(last_hidden, hidden_dim, non_linearity))
        last_hidden = hidden_dim
    layers.append(nn.Linear(last_hidden, output_dim))
    return nn.Sequential(*layers)


