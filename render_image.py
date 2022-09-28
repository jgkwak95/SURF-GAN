import argparse
import math
import numpy as np
import os
import torch
import curriculums
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from PIL import Image
from util import   sample_latent


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--experiment', type=str, default='CelebA_surf')
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--psi', type=float, default=0.7)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA_single')
    parser.add_argument('--specific_ckpt', type=str, default=None)
    parser.add_argument('--depth_map', action='store_true')

    opt = parser.parse_args()


    ## initialize camera parameter
    yaw = math.pi / 2
    pitch = math.pi / 2
    fov = 12

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['h_mean'] = yaw
    curriculum['v_mean'] = pitch
    curriculum['fov'] = fov
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum['feat_dim'] = 512
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}


    if opt.specific_ckpt is not None:
        g_path = f'./{opt.experiment}/{opt.specific_ckpt}'
    else:
        g_path =  f'./{opt.experiment}/generator.pth'

    ### Load
    generator = torch.load(g_path, map_location=torch.device(device))
    ema_file = g_path.split('generator')[0] + 'ema.pth'
    ema_f = torch.load(ema_file)

    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema.load_state_dict(ema_f)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()


    save_dir = f'./result/{opt.experiment}/img'

    seed = opt.seed
    torch.manual_seed(seed)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    zs = sample_latent((1, 9, 6), device=device, truncation=opt.psi)
    z_noise = torch.zeros((1, 1, 256), device=device)



    with torch.no_grad():

        curriculum['h_mean'] = yaw
        curriculum['v_mean'] = pitch
        curriculum['fov'] = fov
        curriculum['h_stddev'] = 0
        curriculum['v_stddev'] = 0

        img = generator.staged_forward(zs, z_noise, **curriculum)[0]
        img = (img.squeeze(0).permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_dir, f"result_{seed}.png"))

###################################################################
####################################################################
