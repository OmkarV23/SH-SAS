import argparse
import torch
import commentjson as json
from inr_reconstruction.utils import divide_chunks
import os
import glob
import numpy as np
import pickle
import constants as c
import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt
import time
import math
from torch.utils.tensorboard import SummaryWriter
from inr_reconstruction.reconstruct_scene_parser import *
from inr_reconstruction.logging_utils import *
from inr_reconstruction.timing_utils import *
from render_utils import set_axes_equal, _set_axes_radius
import pdb
import mcubes
import trimesh
import debugpy

# debugpy.listen(5677)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

def lerp(x, a, b):
    return (b - a) * x + a

def normalize_vertices(vertices, scene_extent, voxel_grid_size):
    min_x = scene_extent[0][0]
    max_x = scene_extent[0][1]
    min_y = scene_extent[1][0]
    max_y = scene_extent[1][1]
    min_z = scene_extent[2][0]
    max_z = scene_extent[2][1]
    num_x = voxel_grid_size[0]
    num_y = voxel_grid_size[1]
    num_z = voxel_grid_size[2]
    
    vertices_temp = np.array(vertices)
    vertices_temp[:, 0] = lerp(vertices[:, 1] / num_y, min_y, max_y)
    vertices_temp[:, 1] = lerp(vertices[:, 0] / num_x, min_x, max_x)
    vertices_temp[:, 2] = lerp(vertices[:, 2] / num_z, min_z, max_z)
    return vertices_temp

def point_cloud_to_mesh_marching_cube(points, experiment_name, output_dir, scene_extent, voxel_grid_size):
    points = np.pad(points, 1)
    points_smooth = mcubes.smooth(points)
    vertices, triangles = mcubes.marching_cubes(points_smooth, 0.0)
    vertices = normalize_vertices(vertices, scene_extent, voxel_grid_size)
    mcubes.export_obj(vertices, triangles, os.path.join(output_dir, f"{experiment_name}mesh.obj"))

with open("./scenes/airsas/bunny_5k/system_data_bunny_5k.pik", 'rb') as handle:
    system_data = pickle.load(handle)

out_dir = './scenes/airsas/bunny_5k/npb_output/bunny_5k_view_sh_7'
experiment_name = 'bunny_5k_view_sh_7'
infer_file = "./scenes/airsas/bunny_5k/npb_output/bunny_5k_view_sh_7/numpy/comp_albedo11000.npy"

all_scene_coords = torch.from_numpy(system_data[c.GEOMETRY][c.VOXELS])
scene_scale_factor = 2

NUM_X = system_data[c.GEOMETRY][c.NUM_X]
NUM_Y = system_data[c.GEOMETRY][c.NUM_Y]
NUM_Z = system_data[c.GEOMETRY][c.NUM_Z]

voxel_size_x = torch.abs(all_scene_coords[:, 0].max() - all_scene_coords[:, 0].min())/NUM_X
voxel_size_y = torch.abs(all_scene_coords[:, 1].max() - all_scene_coords[:, 1].min())/NUM_Y
voxel_size_z = torch.abs(all_scene_coords[:, 2].max() - all_scene_coords[:, 2].min())/NUM_Z

voxel_size_avg = torch.mean(torch.tensor([voxel_size_x, voxel_size_y, voxel_size_z]))
voxel_size_avg = torch.sqrt(voxel_size_avg**2 + voxel_size_avg**2)


voxel_grid_size = (NUM_X, NUM_Y, NUM_Z)
scene_extent = ((all_scene_coords[:,0].min().item(), all_scene_coords[:,0].max().item()),
                (all_scene_coords[:,1].min().item(), all_scene_coords[:,1].max().item()),
                (all_scene_coords[:,2].min().item(), all_scene_coords[:,2].max().item()))


albedo = np.load(infer_file)
albedo = albedo.reshape(NUM_X, NUM_Y, NUM_Z)

# threshold = 0.1 For arma
threshold = 0.1 # to 0.1 (try 0.02, 0.05, 0.1)


mag = np.abs(albedo).astype(float)
mag = (mag - mag.min()) / (mag.max() - mag.min())
condition = mag > threshold

point_cloud_to_mesh_marching_cube(condition, experiment_name, out_dir, scene_extent, voxel_grid_size)