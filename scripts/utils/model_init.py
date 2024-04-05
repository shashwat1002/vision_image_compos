from typing import *
from torch.nn import Module
from torch import nn
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPConfig
from transformers import BlipProcessor, BlipModel, BlipConfig
from transformers import AutoModel, AutoTokenizer, AutoModel, AutoConfig
import torch


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


def init_subject_model(
    model_name: str, model_type: str, model_config=None, device: str = "cpu"
) -> dict:
    """
    To initialize the subject model (the one being studied)
    Inputs:
    * model_name: str: name of the model (as in huggingface)
    * model_type: str: type of the model (Text Encoder or image)
    * model_config: Config object of the model
    Outputs:
    dict: dictionary containing the model and the config and related
    """

    if model_type == "text":
        if model_config is None:
            model_config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, config=model_config)
        model.to(device=device)
        model.eval()
        return {
            "model_text": model,
            "tokenizer": tokenizer,
            "config_text": model_config,
        }
    elif model_type == "clip":
        if model_config is None:
            model_config = CLIPConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name, config=model_config)
        model.to(device=device)
        model.eval()
        model_text = model.text_model
        text_config = model_text.config
        return {
            "model_text": model_text,
            "tokenizer": tokenizer,
            "config_text": text_config,
            "model": model,
            "config": model_config,
            "processor": CLIPProcessor.from_pretrained(model_name, device=device),
        }
    elif model_type == "blip":
        if model_config is None:
            model_config = BlipConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BlipModel.from_pretrained(model_name, config=model_config)
        model.to(device=device)
        model.eval()
        model_text = model.text_model
        text_config = model_text.config
        return {
            "model_text": model_text,
            "tokenizer": tokenizer,
            "config_text": text_config,
            "model": model,
            "config": model_config,
        }
    else:
        raise ValueError("Model type not recognized")
