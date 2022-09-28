import torch
import copy
import os
import torch.distributed as dist
import torch.nn.functional as F





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

def copy_stot(s_param, t_param):

    for s_param, param in zip(s_param.shadow_params, t_param):
        if param.requires_grad:
            param.data.copy_(s_param.data)


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
