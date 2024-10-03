from typing import *
from torch.nn import Module
from torch import nn
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPConfig
from transformers import BlipProcessor, BlipModel, BlipConfig
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModel, AutoConfig
import torch
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from dalle2_pytorch.trainer import DiffusionPriorTrainer
from scripts.utils.diffusion_convenience_inheritence import DiffusionPriorCustom


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


def init_diffusion_prior_model(path: str, device: str = "cpu"):
    prior_network = (
        DiffusionPriorNetwork(
            dim=768,
            depth=24,
            dim_head=64,
            heads=32,
            normformer=True,
            attn_dropout=5e-2,
            ff_dropout=5e-2,
            num_time_embeds=1,
            num_image_embeds=1,
            num_text_embeds=1,
            num_timesteps=1000,
            ff_mult=4,
        )
        .to(device=device)
        .eval()
    )
    clip_model = OpenAIClipAdapter("ViT-L/14")
    diffusion_prior = (
        DiffusionPriorCustom(
            net=prior_network,
            clip=clip_model,
            image_embed_dim=768,
            timesteps=1000,
            cond_drop_prob=0.0,
            loss_type="l2",
            condition_on_text_encodings=True,
        )
        .to(device=device)
        .eval()
    )

    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=1.1e-4,
        wd=6.02e-2,
        max_grad_norm=0.5,
        amp=False,
        group_wd_params=True,
        use_ema=True,
        device="cuda",
        accelerator=None,
    ).eval()

    if path != "random":
        with torch.no_grad():
            loaded_checkpoint = torch.load(path, map_location=device)
            diffusion_prior.load_state_dict(loaded_checkpoint["model"])
    else:
        print("Randomly initialized for control")

    return {
        "diffusion_prior": diffusion_prior,
        "trainer": trainer,
        "prior_network": prior_network,
    }


import clip


def text_embed_prior(text, prior_model, num_samples=2):
    # currently 2 embeds are sampled and the one with smallest cosine distance to text prompt is chosen
    # TODO: add otherways of collating samples
    tokenized_text = clip.tokenize(text).to(prior_model.device)
    predicted_embed = prior_model.sample_embeds(tokenized_text, num_samples=num_samples)
    return predicted_embed


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
        image_config = model.vision_model.config
        return {
            "model_text": model_text,
            "tokenizer": tokenizer,
            "config_text": text_config,
            "model": model,
            "config": model_config,
            "processor": CLIPProcessor.from_pretrained(model_name, device=device),
            "config_image": image_config,
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
    elif model_type == "image":
        if model_config is None:
            model_config = AutoConfig.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name, config=model_config)
        image_model = AutoModel.from_pretrained(model_name, config=model_config)
        image_model.to(device=device)
        image_model.eval()
        return {
            "processor": processor,
            "config_image": model_config,
            "model_image": image_model,
        }
    else:
        raise ValueError("Model type not recognized")
