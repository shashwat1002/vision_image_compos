import clip
from scripts.utils.feature_extraction import (
    get_openai_clip_image_embeds,
)


def loss_eval(
    caption,
    diffusion_prior_trainer,
    clip_model,
    processor,
    image_embed=None,
    image=None,
    num_samples=10,
    device="cpu",
    cond_scale=None,
):
    """
    Evaluate the loss of the model
    :caption: the caption to evaluate (in string)
    :diffusion_prior_trainer: the model to evaluate
    :clip: the clip model
    :image_embed: the image embedding to evaluate (send this either this or image)
    :image: the image to evaluate (send this either this or image_embed)
    """

    if image_embed is None:
        image_embed = get_openai_clip_image_embeds(
            clip_model, processor, image, device=device
        )

    # repeat the image_embed num_samples of times
    image_embed = image_embed.squeeze().unsqueeze(dim=0)
    image_embed = image_embed.repeat(num_samples, 1)

    # tokenize the caption
    caption = clip.tokenize(caption).to(device)

    # print(caption, image_embed.shape)

    return diffusion_prior_trainer(
        text=caption, image_embed=image_embed.clone(), 
    )
