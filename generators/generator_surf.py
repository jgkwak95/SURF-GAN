"""Implicit generator for 3D volumes"""

import random
import torch.nn as nn
import torch
import time
import curriculums
from torch.cuda.amp import autocast

from .volumetric_rendering import *

#---------------------------------------------------------------
class ImplicitGenerator3d_SURF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.siren = SURFSIREN(hidden_dim=256, output_dim=4, input_dim=3, device=None)
        self.epoch = 0
        self.step = 0


    def set_device(self, device):
        self.device = device
        self.siren.device = device

        # self.generate_avg_frequencies()
        # self.generate_avg_w()

    def forward(self, z, z_noise, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean,
                hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z.shape[0]

        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps,
                                                                   resolution=(img_size, img_size), device=self.device,
                                                                   fov=fov, ray_start=ray_start,
                                                                   ray_end=ray_end)  # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(
                points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean,
                device=self.device, mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size,
                                                                                              img_size * img_size * num_steps,
                                                                                              3)
            transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points
        coarse_output = self.siren(transformed_points, z, z_noise,ray_directions=transformed_ray_directions_expanded).reshape(
            batch_size, img_size * img_size, num_steps, 4)

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device,
                                                  clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                         num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(
                    2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(fine_points, z, z_noise, ray_directions=transformed_ray_directions_expanded).reshape(
                batch_size, img_size * img_size, -1, 4)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        # Create images with NeRF
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device,
                                                   white_back=kwargs.get('white_back', False),
                                                   last_back=kwargs.get('last_back', False),
                                                   clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        return pixels, torch.cat([pitch, yaw], -1)

    def staged_forward(self, z, z_noise, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1,
                       lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2,
                       sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z.shape[0]


        with torch.no_grad():


            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps,
                                                                   resolution=(img_size, img_size), device=self.device,
                                                                   fov=fov, ray_start=ray_start,
                                                                   ray_end=ray_end)  # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(
                points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean,
                device=self.device, mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size,
                                                                                              img_size * img_size * num_steps,
                                                                                              3)
            transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # Sequentially evaluate siren with max_batch_size to avoid OOM
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):

                coarse_output[b:b + 1] = self.siren(
                    transformed_points[b:b + 1], z[b:b+1], z_noise[b:b+1],
                    ray_directions=transformed_ray_directions_expanded[b:b + 1])

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)

            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device,
                                                      clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                    #### Start new importance sampling
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             num_steps, det=False).detach().to(self.device)
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(
                        2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,
                                                                                                                    -1,
                                                                                                                    -1,
                                                                                                                    3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                # Sequentially evaluate siren with max_batch_size to avoid OOM
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):

                    fine_output[b:b + 1] = self.siren(
                        fine_points[b:b + 1], z[b:b+1], z_noise[b:b+1],
                        ray_directions=transformed_ray_directions_expanded[b:b + 1])

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)

                all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals

            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device,
                                                       white_back=kwargs.get('white_back', False),
                                                       clamp_mode=kwargs['clamp_mode'],
                                                       last_back=kwargs.get('last_back', False),
                                                       fill_mode=kwargs.get('fill_mode', None),
                                                       noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()

            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        return pixels, depth_map


    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).mean()
                )
        return sum(reg) / len(reg)



class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(30. * x)



def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_output_dim))


        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)

        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]

        return frequencies, phase_shifts

    def get_w(self, z):
        w = self.network[:-1](z)

        return w

    def forward_with_w(self, w):
        frequencies_offsets = self.network[-1](w)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]

        return frequencies, phase_shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)

class SinedLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.sine = Sine()

    def forward(self, x):
        x = self.layer(x)
        x = self.sine(x)
        # freq = freq.unsqueeze(1).expand_as(x)
        # phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return x

class SinedLayer_noise(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.sine = Sine()

    def forward(self, x, noise):
        x = self.layer(x)
        x = x + noise * 0.01
        x = self.sine(x)
        # freq = freq.unsqueeze(1).expand_as(x)
        # phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return x


class SubspaceLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_basis: int,
    ):
        super().__init__()

        self.U = nn.Parameter(torch.empty(n_basis, dim))
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]))
        self.mu = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return (self.L * z) @ self.U + self.mu


class SURFBlock(nn.Module):
    def __init__(
        self,
        feat_dim:int,
        n_basis: int,
        out_dim : int,
        in_dim : int,

    ):
        super().__init__()

        self.feat_dim = feat_dim

        self.projection = SubspaceLayer(dim=feat_dim*2, n_basis=n_basis)
        self.subspace_linear = nn.Linear(feat_dim*2, feat_dim)
        self.feature_linear = nn.Linear(in_dim, out_dim)
        self.sine = Sine()


        self.subspace_linear.apply(frequency_init(25))
        self.feature_linear.apply(frequency_init(25))
        with torch.no_grad():
            self.subspace_linear.weight *= 0.25

    def forward(self, z, h):


        phi = self.projection(z)
        phi = self.subspace_linear(phi)
        h = self.feature_linear(h)

        phi_freq = phi[..., :self.feat_dim // 2]
        phi_phase = phi[..., self.feat_dim // 2:]

        phi_freq = phi_freq * 15 + 30

        phi_freq = torch.unsqueeze(phi_freq, 1).expand_as(h)
        phi_phase = torch.unsqueeze(phi_phase, 1).expand_as(h)

        h = h + torch.sin(phi_freq * h + phi_phase)

        return h




class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor



def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    coordinates = coordinates.float()
    grid = grid.float()

    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=True)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H * W * D, C)
    return sampled_features


def modified_first_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)



#---------------------------------------------
class SURFSIREN(nn.Module):


    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, fourier=False,  device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_basis = 6
        self.n_blocks = 8
        self.fourier = fourier


        self.blocks = nn.ModuleList()



        self.first_layer = SinedLayer_noise(3, hidden_dim)

        for i in range(self.n_blocks):
            self.blocks.append(
                SURFBlock(
                    feat_dim=hidden_dim*2,
                    n_basis=self.n_basis,
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                )
            )

        self.color_layer_sine = SURFBlock(
                                feat_dim=hidden_dim*2,
                                n_basis=self.n_basis,
                                in_dim=hidden_dim+3,
                                out_dim=hidden_dim
        )

        self.color_layer_linear = nn.Linear(hidden_dim, 3)
        self.final_layer = nn.Linear(hidden_dim, 1)


        self.final_layer.apply(frequency_init(25))
        # self.blocks.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.first_layer.apply(first_layer_film_sine_init)

        self.gridwarper = UniformBoxWarp(
            0.24)  # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z_all, noise, ray_directions, **kwargs):


        input = self.gridwarper(input)
        x = input


        zs = z_all[:, :-1, :]
        zn = noise

        x = self.first_layer(x, zn)

        for block, z in zip(self.blocks, zs.permute(1,0,2)):
            x = block(z, x)


        rgb = self.color_layer_sine(z_all[:,self.n_blocks,:], torch.cat([ray_directions,x], dim=-1 ))
        rgb = torch.sigmoid(self.color_layer_linear(rgb))

        sigma = self.final_layer(x)

        return torch.cat([rgb, sigma], dim=-1)

    def sample_latent(self, batch: int, truncation=1.0):

        zs = torch.randn(batch, self.n_blocks, self.n_basis, device=self.device)
        if truncation < 1.0:

            zs = torch.zeros_like(zs) * (1 - truncation) + zs * truncation
        return zs
    def sample(self, batch: int, truncation=1.0, **kwargs):
        return self.forward(self.sample_latent(batch, truncation=truncation), **kwargs)

