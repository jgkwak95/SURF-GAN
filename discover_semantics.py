import argparse
import math
import numpy as np
import os
import torch
import curriculums
from torch_ema import ExponentialMovingAverage
from PIL import Image
from util import  sample_noise, sample_latent, load_ema_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default='CelebA')
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA_single')
    parser.add_argument('--specific_ckpt', type=str, default=None)
    parser.add_argument('--num_id', type=int, default=8)
    parser.add_argument('--intermediate_points', type=int, default=9)
    parser.add_argument('--traverse_range', type=float, default=2.0)
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
    if isinstance(ema_f, dict):
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        load_ema_dict(ema, ema_f)
    else:
        ema = ema_f

    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    save_dir = f'./result/{opt.experiment}/semantic'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    traverse_range = opt.traverse_range
    intermediate_points = opt.intermediate_points
    num_id = opt.num_id

    zs = sample_latent((num_id, 9, 6), device)
    z_noise = sample_noise((num_id, 1, 256), device=device)
    _, n_layers, n_dim = zs.shape

    offsets = np.linspace(-traverse_range, traverse_range, intermediate_points)

    for i_layer in range(n_layers):
        for i_dim in range(n_dim):
            print(f"  layer {i_layer} - dim {i_dim}")
            imgs = []
            for offset in offsets:
                _zs = zs.clone()
                _zs[:, i_layer, i_dim] = offset
                with torch.no_grad():
                    img = generator.staged_forward(_zs, z_noise, **curriculum)[0]
                    img = torch.cat([_img for _img in img], dim=1)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=2)

            imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(imgs).save(
                os.path.join(save_dir, f"traverse_L{i_layer}_D{i_dim}.png")
            )
