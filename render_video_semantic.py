import argparse
import math

from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums
from torch_ema import ExponentialMovingAverage
from util import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--experiment', type=str, default='CelebA')
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA_single')
    parser.add_argument('--specific_ckpt', type=str, default=None)
    parser.add_argument('--traverse_range', type=float, default=2.0)
    parser.add_argument('--L', type=int, required=True)
    parser.add_argument('--D', type=int, required=True)
    parser.add_argument('--num_frames', type=int, default=100)
    parser.add_argument('--mode', type=str, default='circle')
    opt = parser.parse_args()

    ## initialize
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
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    if opt.specific_ckpt is not None:
        g_path = f'./{opt.experiment}/{opt.specific_ckpt}'
    else:
        g_path =  f'./{opt.experiment}/generator.pth'

    ##
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

    save_dir = f'./result/{opt.experiment}/vid'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    traverse_range = opt.traverse_range
    num_frames = opt.num_frames
    mode = opt.mode



############################################################
############################################################
    trajectory = []

    if mode == 'yaw':
        for t in np.linspace(0, 1 , num_frames):

            pitch = math.pi/2
            yaw = -0.7 * np.cos(t * math.pi) + math.pi / 2
            fov = 12
            offset = traverse_range  * np.sin(t * math.pi)
            trajectory.append((pitch, yaw, fov, offset))

    elif mode == 'pitch':
        for t in np.linspace(0, 1 , num_frames):

            pitch = -0.4 * np.cos(t * math.pi) + math.pi / 2
            yaw = math.pi/2
            fov = 12
            offset = traverse_range  * np.sin(t * math.pi)
            trajectory.append((pitch, yaw, fov, offset))

    elif mode == 'yaw_pitch':
        for t in np.linspace(0, 1 , num_frames):

            pitch = 0.4 * np.sin( t  * math.pi) + math.pi / 2
            yaw = -0.7 * np.cos(t * math.pi) + math.pi / 2
            fov = 12
            offset = traverse_range  * np.sin(t * math.pi)
            trajectory.append((pitch, yaw, fov, offset))

    elif mode == 'fov':
        for t in np.linspace(0, 1 , num_frames):

            pitch = math.pi/2
            yaw = math.pi/2
            fov = 9 + t * 6
            offset = traverse_range  * np.sin(t * math.pi)
            trajectory.append((pitch, yaw, fov, offset))

    elif mode == 'static':
        for t in np.linspace(0, 1 , num_frames):

            pitch = math.pi/2
            yaw = math.pi/2
            fov = 12

            offset = traverse_range  * np.sin(t * math.pi)
            trajectory.append((pitch, yaw, fov, offset))

    elif mode == 'circle':
        for t in np.linspace(0, 1, num_frames):
            pitch = 0.2 * np.cos(t * 4 * math.pi) + math.pi / 2
            yaw = 0.35 * np.sin(t * 4*  math.pi) + math.pi / 2
            fov = 12

            # offset = traverse_range / 2 + np.abs(traverse_range * (t - 0.5))
            offset = traverse_range  * np.sin(t * math.pi)

            trajectory.append((pitch, yaw, fov, offset))

    else:
        raise Exception("You should choose a mode")

############################################################
########################################################

    seed = opt.seed
    torch.manual_seed(seed)

    zs = sample_latent((1, 9, 6), device=device)
    z_noise = sample_noise((1, 1, 256), device=device)
    _, n_layers, n_dim = zs.shape


    # Target Layer and Dim (L#D#)
    ####
    i_layer = opt.L
    i_dim = opt.D
    #####

    frames = []
    depths = []

    output_name = f'{seed}_{mode}_L{i_layer}D{i_dim}.mp4'
    writer = skvideo.io.FFmpegWriter(os.path.join(save_dir, output_name),
                                     outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})


    with torch.no_grad():
        for pitch, yaw, fov, offset in tqdm(trajectory):
            curriculum['h_mean'] = yaw
            curriculum['v_mean'] = pitch
            curriculum['fov'] = fov
            curriculum['h_stddev'] = 0
            curriculum['v_stddev'] = 0
            _zs = zs.clone()
            _zs[:, i_layer, i_dim] = offset
            frame = generator.staged_forward(_zs, z_noise, **curriculum)[0]
            frames.append(tensor_to_PIL(frame))

        for frame in frames:
            writer.writeFrame(np.array(frame))

        writer.close()
