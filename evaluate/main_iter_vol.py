import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pickle
import constants as c
import argparse


def matplotlib_render(mag, thresh, x_voxels, y_voxels, z_voxels, save_path):
    mag = np.abs(mag)
    mag = mag.ravel()

    u = mag.mean()
    var = mag.std()
    mag[mag[:] < (u + thresh * var)] = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    im = ax.scatter(x_voxels,
                    y_voxels,
                    z_voxels,
                    c=mag, alpha=0.5)
    ax.set_xlim3d(
        (x_voxels.min(), x_voxels.max()))
    ax.set_ylim3d(
        (y_voxels.min(), y_voxels.max()))
    ax.set_zlim3d(
        (z_voxels.min(), z_voxels.max()))
    plt.grid(True)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    fig.savefig(save_path)
    plt.close(fig)
    return fig

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Export 3D point cloud figure")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--system_data_path', required=True, help="System data path")
    
    parser.add_argument('--object', required=True, help="Experiment name")
    parser.add_argument('--comp_albedo_path', required=True, help="Complex albedo path")
    parser.add_argument('--thresh', type=float, default=2.0, help="Threshold for reconstructed inr")

    args = parser.parse_args()

    with open(args.system_data_path, 'rb') as handle:
        system_data = pickle.load(handle)

    corners = system_data[c.GEOMETRY][c.CORNERS]
    all_scene_coords = system_data[c.GEOMETRY][c.VOXELS]

    comp_albedo = np.load(args.comp_albedo_path)   # load calculated comp_albedo

    iteration = args.comp_albedo_path.split('comp_albedo')[-1].split('.')[0]

    matplotlib_render(comp_albedo, args.thresh,
                      all_scene_coords[:, 0], all_scene_coords[:, 1], all_scene_coords[:, 2],
                      f"{args.output_dir}/{args.object}_comp_albedo_{iteration}.png")

