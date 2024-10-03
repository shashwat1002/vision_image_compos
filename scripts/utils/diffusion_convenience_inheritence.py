from dalle2_pytorch import DiffusionPrior
import torch
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def l2norm(t):
    return F.normalize(t, dim=-1)


class DiffusionPriorCustom(DiffusionPrior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def p_losses(self, image_embed, times, text_cond, noise=None, cond_scale=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.q_sample(x_start=image_embed, t=times, noise=noise)

        if cond_scale is None:
            pred = self.net(
                image_embed_noisy,
                times,
                cond_drop_prob=self.cond_drop_prob,
                **text_cond
            )
        else:
            pred = self.net.forward_with_cond_scale(
                image_embed_noisy, times, cond_scale=cond_scale, **text_cond
            )

        if self.predict_x_start and self.training_clamp_l2norm:
            print("hi")
        print(self.predict_x_start, self.training_clamp_l2norm)
        pred = l2norm(pred) * self.image_embed_scale

        # target = noise if not self.predict_x_start else image_embed
        target = l2norm(image_embed) * self.image_embed_scale

        loss = self.loss_fn(pred, target)
        return loss
