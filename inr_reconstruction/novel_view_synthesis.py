import argparse
import torch
import commentjson as json
from utils import divide_chunks
import os
import glob
import numpy as np
import pickle
import constants as c
from utils import aggressive_crop_weights, normalize
from sas_utils import hilbert_torch, figure_to_tensorboard, safe_normalize, matplotlib_render, range_normalize, find_indeces_within_scene   
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import shutil
from argument_io import directory_cleaup
from forward_model import scattering_model, transmission_model, render_transmittance_from_density, render_transmittance_from_alpha
from sampling import SceneSampler, normalize_vectors
import time
import math
from torch.utils.tensorboard import SummaryWriter
from reconstruct_scene_parser import *
from logging_utils import *
from timing_utils import *
from render_utils import set_axes_equal, _set_axes_radius
import pdb


if __name__ == '__main__':
    ###################################
    # (0) torch initialization
    ###################################
    torch.autograd.set_detect_anomaly(True)
    assert torch.cuda.is_available()
    dev = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    logger = load_logger("Info logger")

    ###################################
    # (1) parse from config
    ###################################

    parser = config_parser()
    args = parser.parse_args()

    num_rays = int(math.sqrt(args.num_rays)) ** 2

    num_importance_sampling_rays = None
    if args.importance_sampling_rays is not None:
        num_importance_sampling_rays = int(math.sqrt(args.importance_sampling_rays)) ** 2

    if num_importance_sampling_rays is not None:
        logger.info("Using %d sampling rays and %d importance sampling rays" % (num_rays,
                                                                                num_importance_sampling_rays))
    else:
        logger.info(
            "Using %d sampling rays and 0 importance sampling rays" % (num_rays))
    logger.info("Beamwidth is %s" % str(args.beamwidth))

    # assert few properties
    assert args.sampling_distribution_uniformity <= 1.00001
    assert args.sampling_distribution_uniformity >= 0.
    if args.k_normal_per_ray is not None:
        assert args.k_normal_per_ray == 1, "Only support 1 normal per ray for now"

    with open(args.scene_inr_config) as config_file:
        inr_config = json.load(config_file)

    # only runs with --clear_output_directory flag

    # make directories for output
    expname = args.expname
    if expname is None:
        raise OSError("Failed to read the --expname argument")
        expname = os.getcwd()
        expname = expname.split("/")[-1]
    basedir = os.path.join(args.output_dir, expname)

    ###################################
    # (2) Load data and process
    ###################################

    with open(args.system_data, 'rb') as handle:
        system_data = pickle.load(handle)

    corners = torch.from_numpy(system_data[c.GEOMETRY][c.CORNERS]).to(dev)
    if args.no_factor_4:
        corners_scaled = corners
    else:
        corners_scaled = 1.5*(corners + .05)

    all_scene_coords = torch.from_numpy(system_data[c.GEOMETRY][c.VOXELS])
    scene_scale_factor = 1
    if args.no_factor_4:
        scene_scale_factor = 1 / (all_scene_coords.abs().max())
    else:
        scene_scale_factor = 1 / (4 * all_scene_coords.abs().max())
    if args.normalize_scene_dims:
        all_scene_coords = all_scene_coords * scene_scale_factor
        corners_scaled = corners_scaled * scene_scale_factor
        #corners = torch.from_numpy(corners).to(dev) * scene_scale_factor

    NUM_X = system_data[c.GEOMETRY][c.NUM_X]
    NUM_Y = system_data[c.GEOMETRY][c.NUM_Y]
    NUM_Z = system_data[c.GEOMETRY][c.NUM_Z]

    voxel_size_x = torch.abs(all_scene_coords[:, 0].max() - all_scene_coords[:, 0].min())/NUM_X
    voxel_size_y = torch.abs(all_scene_coords[:, 1].max() - all_scene_coords[:, 1].min())/NUM_Y
    voxel_size_z = torch.abs(all_scene_coords[:, 2].max() - all_scene_coords[:, 2].min())/NUM_Z

    voxel_size_avg = torch.mean(torch.tensor([voxel_size_x, voxel_size_y, voxel_size_z]))
    voxel_size_avg = torch.sqrt(voxel_size_avg**2 + voxel_size_avg**2)


    scene_voxels=None

    try:
        tx_bw = torch.tensor([system_data[c.SYS_PARAMS][c.TX_BW]]).to(dev)
        assert args.bw_units is not None, "Detected a set TX beamwidth in system_data file. " \
                                            "Specifify with --bw_units whether this is degrees or radians"
        if args.bw_units == 'r':
            tx_bw = torch.rad2deg(tx_bw)

    except AssertionError:
        tx_bw = None
    except KeyError:
        tx_bw = None

    # override the system file
    if args.beamwidth is not None:
        tx_bw = torch.tensor([args.beamwidth])

    if args.two_dimensions:
        # Assert z value of corners is the same if want to only use two dimensions
        assert (corners[..., 2][0] == corners[..., 2][1:]).all()

    # The exception supports older code functionality
    try:
        speed_of_sound = system_data[c.SOUND_SPEED]
    except KeyError:
        temp = np.mean(system_data[c.TEMPS])
        speed_of_sound = 331.4 + 0.6 * temp

    weight_paths = glob.glob(os.path.join(args.fit_folder, c.NUMPY, c.WEIGHT_PREFIX + '*'))

    # Older das.sh code puts in different directory, so check:
    if len(weight_paths) == 0:
        weight_paths = glob.glob(os.path.join(args.fit_folder, c.NUMPY, c.WEIGHT_PREFIX + '*'))

    assert len(weight_paths) > 0, "Failed to load weights from " + \
                                    os.path.join(args.fit_folder, c.NUMPY, c.WEIGHT_PREFIX + '*')

    wfm_crop_settings = system_data[c.WFM_CROP_SETTINGS]
    # Sort weight paths so they align with tx indeces
    num_radial = None
    if args.use_mf_wfms:
        weights = system_data[c.WFM_RC]
        if args.use_up_to is not None:
            weights = weights[:args.use_up_to*360, :]
        weights = weights[:, wfm_crop_settings[c.MIN_SAMPLE]:
                            wfm_crop_settings[c.MIN_SAMPLE] + wfm_crop_settings[c.NUM_SAMPLES]]
        print("Using MF wfms", weights.shape)
        num_radial = weights.shape[-1]

    else:
        weight_paths = sorted(weight_paths, key=lambda x: int(x.split('_')[-2]))
        if args.use_up_to is not None:
            weight_paths = weight_paths[0:args.use_up_to]

        # Store all weights in RAM?
        weights = []

        for index in tqdm(range(len(weight_paths)), desc="Reading weight paths INR"):
            weight_path = weight_paths[index]
            split_weight_path = weight_path.split('_')
            start_index = int(split_weight_path[-2])
            stop_index = int(split_weight_path[-1].split('.')[0]) + 1
            weight = np.load(weight_path)
            num_radial = weight.shape[-1]

            weights.append(np.load(weight_path))

        weights = np.concatenate((weights), axis=0)

        # Convert weight to complex analytic signal
        weights = torch.from_numpy(weights)
        comp_weights = torch.zeros_like(weights, dtype=torch.complex64)
        for i in tqdm(range(weights.shape[0]), desc="Converting weights to complex"):
            comp_weights[i, :] = hilbert_torch(weights[i, :].float())

        weights = comp_weights.detach().cpu().numpy()
        logger.info("Loaded deconvolved weights %s" % str(weights.shape))

    new_min_sample, new_max_sample = aggressive_crop_weights(
        tx_coords=torch.from_numpy(system_data[c.TX_COORDS]).to(dev),
        rx_coords=torch.from_numpy(system_data[c.RX_COORDS]).to(dev),
        corners=corners,
        old_min_dist=wfm_crop_settings[c.MIN_DIST],
        old_max_dist=wfm_crop_settings[c.MAX_DIST],
        num_radial=num_radial)

    weights = weights[:, new_min_sample:new_max_sample]
    print("Weights", weights.shape)
    distribution = np.abs(weights)**(args.sampling_distribution_uniformity/1) / \
                    np.sum(np.abs(weights)**(args.sampling_distribution_uniformity/1), axis=-1)[..., None]

    indeces = np.arange(0, weights.shape[-1], 1)

    dists_norm = torch.linspace(0, 1, num_radial*args.upsample)
    dists_scene = wfm_crop_settings[c.MIN_DIST] + \
                    dists_norm * (wfm_crop_settings[c.MAX_DIST] - wfm_crop_settings[c.MIN_DIST])
    dists_scene = dists_scene[new_min_sample*args.upsample:new_max_sample*args.upsample].to(dev)

    min_delta_dist = torch.abs(dists_scene[0] - dists_scene[1])/2

    perturb_radii = None
    if args.perturb_radii:
        perturb_radii = min_delta_dist

###################################
# (3) Create network and load weights
###################################

start = 0

if args.model_type == 'sh_mlp':
    # Create a network
    from network_sh_mlp import Network
    scene_model = Network(mlp_dim=args.num_neurons,
                        mlp_num_layers=args.num_layers,
                        num_channels=2,
                        device=dev,
                        max_SH_degree=args.sh_levels)
elif args.model_type == 'nvrcsas':
    from network import Network
    scene_model = Network(inr_config=inr_config,
                            dev=dev,
                            num_layers=4,
                            num_neurons=128,
                            scene_voxels=scene_voxels,
                            incoherent=args.incoherent,
                              real_only=args.real_only)
else:
    raise OSError("Unknown model type %s" % args.model_type)

# Load checkpoints
if args.ft_path is not None and args.ft_path != 'None':
    ckpts = [args.ft_path]
    ckpt = torch.load(ckpts[0])
    scene_model.load_state_dict(ckpt['network_fn_state_dict'])
else:
    raise OSError("Failed to load a checkpoint. Specify with --ft_path")

logger.info('Found ckpts: %s' % str(ckpts))
        

random.seed(0)
global_step = start

#assert tx_bw is not None

###################################
# (4) Sampling
###################################

# Create Sampler
# If beamwidth is not none, then this will compute the sparse rays.
scene_sampler = SceneSampler(num_dense_rays=num_rays,
                                num_sparse_rays=num_importance_sampling_rays,
                                max_distance=wfm_crop_settings[c.MAX_DIST],
                                beamwidth=tx_bw,
                                device=dev)

if args.real_only:
    weights = np.real(weights)

if args.incoherent:
    weights = np.abs(weights)

#pdb.set_trace()

dists_scene_cropped = None
ts = []

# if args.sequential_sample:
#     rand_batch = global_step % weights.shape[0]
# else:
#     rand_batch = random.sample(range(weights.shape[0]), 1)

rand_batch = args.rand_batch

weight_batch = weights[rand_batch].squeeze()
dist_batch = distribution[rand_batch].squeeze()

if args.max_weights < dist_batch.shape[-1]:
    index_batch = np.sort(np.random.choice(indeces, size=args.max_weights, replace=False, p=dist_batch))
else:
    index_batch = np.arange(0, dist_batch.shape[-1])

gt_weight_cropped = torch.from_numpy(weight_batch[index_batch]).to(dev) * args.scale_factor


dists_scene_cropped_before = dists_scene[index_batch]

if args.perturb_sampling:
    perturb = torch.from_numpy(np.random.uniform(low=-1, high=1,
                                                    size=dists_scene_cropped_before.shape)).\
        to(dists_scene_cropped_before.device)
    dists_scene_cropped = dists_scene_cropped_before + min_delta_dist*perturb
else:
    dists_scene_cropped = dists_scene_cropped_before


tx_batch = torch.from_numpy(system_data[c.TX_COORDS][rand_batch])
rx_batch = torch.from_numpy(system_data[c.RX_COORDS][rand_batch])

# Exceptions are to support older versions of das.sh that don't set this key
try:
    tx_vec = torch.from_numpy(system_data[c.TX_VECS][rand_batch]).to(dev).squeeze()
except AssertionError:
    tx_vec = None
except KeyError:
    tx_vec = None

if args.flip_z:
    assert tx_vec.ndim == 1
    tx_vec[-1] = -tx_vec[-1]

sampling_distribution=None

debug_dir = None
compute_normals = True

vec_to, dir_to, model_out = scene_sampler.ellipsoidal_sampling(
    radii=dists_scene_cropped,
    tx_pos=tx_batch.to(dev),
    rx_pos=rx_batch.to(dev),
    num_rays=num_rays,
    scene_bounds=corners,
    tx_vec=tx_vec,
    create_return_vec=args.ray_trace_return,
    point_at_center=args.point_at_center,
    distribution=sampling_distribution,
    debug_dir=debug_dir,
    transmit_from_tx=args.transmit_from_tx,
    scene_model=scene_model,
    transmission_model=render_transmittance_from_density,
    occlusion_scale=args.occlusion_scale,
    compute_normals=compute_normals,
    scene_scale_factor=scene_scale_factor,
    device=dev)

if args.normalize_scene_dims:
    vec_to = vec_to * scene_scale_factor

num_to_rad, num_to_rays, _ = vec_to.shape


###################################
# (5) Render
###################################

dir_in = normalize_vectors((dir_to))

start_ev = torch.cuda.Event(enable_timing=True)
end_ev   = torch.cuda.Event(enable_timing=True)

start_ev.record()

if args.model_type == 'sh_mlp':
    model_out = scene_model(positions=vec_to.reshape(-1,3),
                            rx_pos=rx_batch.to(dev).reshape(-1, 3).float())

    scatterers_to = model_out['scatterers_to']
    sigma = model_out['sigma']
    normals = model_out['normals']

    scatterers_to = scatterers_to.reshape(num_to_rad, num_to_rays)
    sigma = sigma.reshape(num_to_rad, num_to_rays)
    if compute_normals:
        normals = normals.reshape(num_to_rad, num_to_rays, 3)


    transmission_probs, alpha = render_transmittance_from_density(radii=dists_scene_cropped,
                                            scatterers_to=sigma,
                                            occlusion_scale=args.occlusion_scale,
                                            factor2=True)
    
elif args.model_type == 'nvrcsas':
    model_out = scene_model(coords_to=vec_to.reshape(-1, 3), compute_normals=compute_normals)
    scatterers_to = model_out['scatterers_to']
    normals = model_out['normals']

    scatterers_to = scatterers_to.reshape(num_to_rad, num_to_rays)
    if compute_normals:
        normals = normals.reshape(num_to_rad, num_to_rays, 3)

    transmission_probs = transmission_model(radii=dists_scene_cropped,
                                            scatterers_to=scatterers_to,
                                            occlusion_scale=args.occlusion_scale,
                                            factor2=True)[0]
else:
    raise OSError("Unknown model type %s" % args.model_type)

lambertian = torch.sum((normals*(-dir_in[None, ...])), dim=-1).clamp(min=0)
scatterers_to_integrate = scatterers_to * transmission_probs * lambertian
estimated_weights = torch.sum(scatterers_to_integrate, dim=-1)

end_ev.record()
torch.cuda.synchronize()
elapsed_ms = start_ev.elapsed_time(end_ev)
print(f"Render GPU time: {elapsed_ms:.2f} ms")


l1_real = torch.nn.functional.l1_loss(estimated_weights.real.float().squeeze(),
                                            gt_weight_cropped.real.float().squeeze(), reduction='mean') 
l1_imag = torch.nn.functional.l1_loss(estimated_weights.imag.float().squeeze(),
                                            gt_weight_cropped.imag.float().squeeze(), reduction='mean')
l1_abs = torch.nn.functional.l1_loss(estimated_weights.abs().float().squeeze(),
                                            gt_weight_cropped.abs().float().squeeze(), reduction='mean')

print("L1 loss (real): ", l1_real.item())
print("L1 loss (imag): ", l1_imag.item())
print("L1 loss (abs): ", l1_abs.item())

mse_real = torch.nn.functional.mse_loss(estimated_weights.real.float().squeeze(),
                                            gt_weight_cropped.real.float().squeeze(), reduction='mean')
mse_imag = torch.nn.functional.mse_loss(estimated_weights.imag.float().squeeze(),
                                            gt_weight_cropped.imag.float().squeeze(), reduction='mean')
mse_abs = torch.nn.functional.mse_loss(estimated_weights.abs().float().squeeze(),
                                            gt_weight_cropped.abs().float().squeeze(), reduction='mean')
print("MSE loss (real): ", mse_real.item())
print("MSE loss (imag): ", mse_imag.item())
print("MSE loss (abs): ", mse_abs.item())


x = dists_scene_cropped.detach().cpu().numpy()

# Real
plt.figure()
plt.plot(x, estimated_weights.real.detach().cpu().numpy(), color='blue',    label='Estimated (real)')
plt.plot(x, gt_weight_cropped.real.cpu().numpy(),          color='orange', label='Ground truth (real)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(basedir, f'{args.model_type}_weights_comparison_real.png'))

# Imag
plt.figure()
plt.plot(x, estimated_weights.imag.detach().cpu().numpy(), color='blue',    label='Estimated (imag)')
plt.plot(x, gt_weight_cropped.imag.cpu().numpy(),          color='orange', label='Ground truth (imag)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(basedir, f'{args.model_type}_weights_comparison_imag.png'))

# Abs
plt.figure()
plt.plot(x, estimated_weights.abs().detach().cpu().numpy(), color='blue',    label='Estimated (abs)')
plt.plot(x, gt_weight_cropped.abs().cpu().numpy(),          color='orange', label='Ground truth (abs)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(basedir, f'{args.model_type}_weights_comparison_abs.png'))
