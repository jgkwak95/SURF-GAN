import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from util import setup, cleanup, sample_latent, sample_noise
from generators import generator_surf
from generators.generator_surf import *

from discriminators import sgdiscriminators


import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage


def train(rank, world_size, opt):


    setup(rank, world_size, opt.port)
    device = torch.device(rank)


    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)


    fixed_z = sample_latent((25, 9, metadata["n_basis"]), device=device)
    fixed_noise = sample_noise((25, 1, 256), device=device)

    CHANNELS = 3

    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != '':
        generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)

        ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
    else:
        generator = getattr(generator_surf, metadata['generator'])().to(device)
        discriminator = getattr(sgdiscriminators, metadata['discriminator'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module



    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*5e-2}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth')))
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth')))
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scaler.pth')))

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    for _ in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        if not dataloader or dataloader.batch_size != metadata['batch_size']:

            dataloader, CHANNELS = datasets.get_dataset_distributed(metadata['dataset'],
                                        world_size,
                                        rank,
                                        **metadata)


            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)


            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        for i, (imgs, _) in enumerate(dataloader):
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                torch.save(ema, os.path.join(opt.output_dir, str(discriminator.step)+'_'+str(metadata['img_size']) + '_ema.pth'))
                torch.save(ema2, os.path.join(opt.output_dir, str(discriminator.step)+'_'+str(metadata['img_size']) + '_ema2.pth'))
                torch.save(generator_ddp.module, os.path.join(opt.output_dir, str(discriminator.step)+'_'+str(metadata['img_size']) + '_generator.pth'))
                torch.save(discriminator_ddp.module, os.path.join(opt.output_dir, str(discriminator.step)+'_'+str(metadata['img_size']) + '_discriminator.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, str(discriminator.step)+'_'+str(metadata['img_size']) + '_optimizer_G.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, str(discriminator.step)+'_'+str(metadata['img_size']) + '_optimizer_D.pth'))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, str(discriminator.step)+'_'+str(metadata['img_size']) + '_scaler.pth'))
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata['batch_size'] : break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z = sample_latent((real_imgs.shape[0], 9, metadata["n_basis"]), device=device)
                    z_noise = sample_noise((real_imgs.shape[0], 1, 256), device=device)
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        subset_z_noise = z_noise[split * split_batch_size:(split + 1) * split_batch_size]
                        g_imgs, g_pos = generator_ddp(subset_z, subset_z_noise, **metadata)

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                # r_preds, _, _ = discriminator_ddp(real_imgs, alpha, **metadata)
                r_preds,  _ = discriminator_ddp(real_imgs, alpha, **metadata)
            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:

                    position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = position_penalty
                else:
                    identity_penalty=0

                d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)


            # TRAIN GENERATOR
            z = sample_latent((imgs.shape[0], 9, metadata["n_basis"]), device=device)
            z_noise = sample_noise((real_imgs.shape[0],1, 256), device=device)
            split_batch_size = z.shape[0] // metadata['batch_split']

            for split in range(metadata['batch_split']):

                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    subset_z_noise = z_noise[split * split_batch_size:(split + 1) * split_batch_size]
                    gen_imgs, gen_positions = generator_ddp(subset_z, subset_z_noise, **metadata)
                    g_preds, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                    g_preds = torch.topk(g_preds, topk_num, dim=0).values

                    if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:

                        position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_penalty =  position_penalty
                    else:
                        identity_penalty = 0

                    g_loss_reg = generator_ddp.module.orthogonal_regularizer()
                    g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty + g_loss_reg * 100
                    generator_losses.append(g_loss.item())

                scaler.scale(g_loss).backward()
                # scaler.scale(g_loss).backward(retain_graph=True)





            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())

            # print(os.environ['CUDA_VISIBLE_DEVICES'])
            if rank == 0:
                interior_step_bar.update(1)
                if i%20 == 0:

                    tqdm.write( f"[Experiment: {opt.output_dir}] [D loss: {d_loss.item():.2f}] [G loss: {g_loss.item():.2f}] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale():.2f}]")
                if discriminator.step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device), fixed_noise.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),fixed_noise.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_tilted.png"), nrow=5, normalize=True)

                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),fixed_noise.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),fixed_noise.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_tilted_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['img_size'] = 128
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            gen_imgs = generator_ddp.module.staged_forward(torch.randn_like(fixed_z).to(device), fixed_noise.to(device), **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_random.png"), nrow=5, normalize=True)

                    ema.restore(generator_ddp.parameters())

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator_ddp.module, os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_ddp.module, os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

                if rank == 0:
                    fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, target_size=metadata['img_size'], data_path=metadata['dataset_path'])
                    # fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir,
                    #                                 **metadata)
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(generator_ddp, metadata, rank, world_size, generated_dir)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=metadata['img_size'])

                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step+1}:{fid}')

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='CelebA')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, default='CelebA_single')
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    # num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    num_gpus = 1 # single
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
