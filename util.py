import torch
import copy
import os
import torch.distributed as dist
import torch.nn.functional as F



def copy_stot(s_param, t_param):

    for s_param, param in zip(s_param.shadow_params, t_param):
        if param.requires_grad:
            param.data.copy_(s_param.data)


def load_ema_dict(ema, state_dict: dict) -> None:
    r"""Loads the ExponentialMovingAverage state.

    Args:
        state_dict (dict): EMA state. Should be an object returned
            from a call to :meth:`state_dict`.
    """
    # deepcopy, to be consistent with module API
    state_dict = copy.deepcopy(state_dict)
    decay = state_dict["decay"]
    if decay < 0.0 or decay > 1.0:
        raise ValueError('Decay must be between 0 and 1')
    ema.num_updates = state_dict["num_updates"]
    assert ema.num_updates is None or isinstance(ema.num_updates, int), \
        "Invalid num_updates"

    ema.shadow_params = state_dict["shadow_params"]
    assert isinstance(ema.shadow_params, list), \
        "shadow_params must be a list"
    assert all(
        isinstance(p, torch.Tensor) for p in ema.shadow_params
    ), "shadow_params must all be Tensors"

    ema.collected_params = state_dict["collected_params"]
    if ema.collected_params is not None:
        assert isinstance(ema.collected_params, list), \
            "collected_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in ema.collected_params
        ), "collected_params must all be Tensors"
        assert len(ema.collected_params) == len(ema.shadow_params), \
            "collected_params and shadow_params had different lengths"


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images

def sample_latent(shape,  device, truncation=1.0):

    zs = torch.randn(shape, device=device)
    if truncation < 1.0:
        zs = torch.zeros_like(zs) * (1 - truncation) + zs * truncation
    return zs


def sample_noise(shape,  device, truncation=1.0):

    zs = torch.randn(shape, device=device)
    if truncation < 1.0:
        zs = torch.zeros_like(zs) * (1 - truncation) + zs * truncation

    return zs