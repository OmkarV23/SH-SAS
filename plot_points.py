import numpy as np
import pyvista as pv

# ------------------------------------------------------------------
# Load data
vec1 = np.load("./scenes/airsas/arma_20k/selected_points.npy")
vec2 = np.load("./scenes/airsas/arma_20k/all_points.npy")

# World bounds tensor: (xmin, ymin, zmin, xmax, ymax, zmax)
world_bounds = np.array([-0.1250, -0.1250, 0.0000,
                          0.1250,  0.1250, 0.2000])

# Re‑order to (xmin, xmax, ymin, ymax, zmin, zmax) for pyvista
bounds = (
    world_bounds[0], world_bounds[3],   # x‑min, x‑max
    world_bounds[1], world_bounds[4],   # y‑min, y‑max
    world_bounds[2], world_bounds[5],   # z‑min, z‑max
)
# ------------------------------------------------------------------
# Build the scene
plotter = pv.Plotter()
plotter.set_background('white')

# # Point clouds
plotter.add_points(pv.PolyData(vec1), color='red',  point_size=10, render_points_as_spheres=True)
# plotter.add_points(pv.PolyData(vec2), color='blue', point_size=5,  render_points_as_spheres=True)

# Bounding box (wireframe cube)
bbox = pv.Box(bounds=bounds)
plotter.add_mesh(bbox, style='wireframe', color='black', line_width=2, opacity=1.0)

# Optional: show axes triad for orientation
plotter.add_axes(line_width=2, color='black', x_color='red', y_color='green', z_color='blue')

# Display
plotter.show()
