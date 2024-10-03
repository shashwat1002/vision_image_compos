import torch


def get_text_feature_across_layers_clip(model_dict, text, pooling="pooler"):
    """
    Get the text features across layers
    Inputs:
    * model_text: torch.nn.Module: the text model
    * text: str: the text
    * pooling: str: the pooling strategy
    Outputs:
    List[torch.Tensor]: the text features across layers
    """
    model = model_dict["model"]
    model_text = model_dict["model_text"]
    tokenizer = model_dict["tokenizer"]

    text_projection = model.text_projection
    device = model.device
    eos_token = model_text.config.eos_token_id

    input_text = tokenizer(text, return_tensors="pt").to(device=device)
    eos_token_index = (input_text["input_ids"] == eos_token).int().argmax(dim=-1)

    text_features_post_proj = []
    text_features_pre_proj = []

    with torch.no_grad():
        hiddens = model_text(**input_text, output_hidden_states=True).hidden_states
        if pooling == "pooler":
            text_features_pre_proj = [
                model_text.final_layer_norm(embds)[:, eos_token_index, :].squeeze()
                for embds in hiddens
            ]

            text_features_post_proj = [
                text_projection(embds).squeeze() for embds in text_features_pre_proj
            ]

        if pooling == "mean":
            text_features_pre_proj = [
                model_text.final_layer_norm(embds).mean(dim=1).squeeze()
                for embds in hiddens
            ]

            text_features_post_proj = [
                text_projection(embds).squeeze() for embds in text_features_pre_proj
            ]

    return {
        "text_features_pre_proj": text_features_pre_proj,
        "text_features_post_proj": text_features_post_proj,
    }


def get_embedding_wino_eval(
    model_dict, c1, c2, i1, i2, proj: bool = True, pooling: str = "pooler"
):
    processor = model_dict["processor"]
    tokenizer = model_dict["tokenizer"]
    model_text = model_dict["model_text"]
    model_image = model_dict["model"].vision_model
    device = model_dict["model"].device
    model = model_dict["model"]

    input_image_1 = processor(images=i1, return_tensors="pt", padding=True).to(
        device=device
    )
    input_image_2 = processor(images=i2, return_tensors="pt", padding=True).to(
        device=device
    )
    embedding_key = ""
    if proj:
        embedding_key = "text_features_post_proj"
    else:
        embedding_key = "text_features_pre_proj"

    image_projection = model.visual_projection

    with torch.no_grad():
        # print(model_image(**input_image_1).last_hidden_state.shape)
        image_1_features = image_projection(model_image(**input_image_1).pooler_output)
        image_2_features = image_projection(model_image(**input_image_2).pooler_output)
        print(image_2_features.shape)
        # image_2_features
        text_1_features, text_2_features = [], []

        text_1_features = get_text_feature_across_layers_clip(
            model_dict=model_dict, text=c1, pooling=pooling
        )[embedding_key]

        text_2_features = get_text_feature_across_layers_clip(
            model_dict=model_dict, text=c2, pooling=pooling
        )[embedding_key]

    return text_1_features, text_2_features, image_1_features, image_2_features


def get_openai_clip_image_embeds(model, preprocess, i, device="cuda"):
    # function that runs a pass of the openai clip library
    import clip

    try:
        model_clip = model.clip
    except AttributeError:
        model_clip = model
    processed = preprocess(i).to(device).unsqueeze(0)
    image = model_clip.encode_image(processed).unsqueeze(0)
    return image


from einops import repeat


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def sample_prior_embeddings(
    prior_model,
    text,
    num_samples_per_batch=10,
    device="cuda",
    timesteps=None,
    cond_scale=1.0,
):
    timesteps = default(timesteps, prior_model.num_timesteps)

    # copied from the sample function in the prior model
    text = repeat(text, "b ... -> (b r) ...", r=num_samples_per_batch)

    batch_size = text.shape[0]
    image_embed_dim = prior_model.image_embed_dim

    text_embed, text_encodings, text_mask = prior_model.clip.embed_text(text)
    # text_embed = text_embed.to(device)
    # text_encodings = text_encodings.to(device)

    text_cond = dict(text_embed=text_embed)

    # if prior_model.condition_on_text_encodings:
    #     text_cond = {**text_cond, "text_encodings": text_encodings}

    image_embeds = prior_model.p_sample_loop(
        (batch_size, image_embed_dim),
        text_cond=text_cond,
        cond_scale=cond_scale,
    )

    return image_embeds
