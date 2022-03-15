import os
import sys
import torch
import pytorch3d
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.io import IO

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from tqdm import tqdm

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
)

from plot_image_grid import image_grid

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
object_id = 6
#DATA_DIR = "./data"
DATA_DIR = "/hri/localdisk/yjin/intro_ros_ws/src/intro_object_models/models/ycb-video"

obj_filename = os.path.join(DATA_DIR, "%03d/textured_simple.obj"%object_id)
#obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
# to its original center and scale.  Note that normalizing the target mesh,
# speeds up the optimization but is not necessary!
verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)))


# the number of different viewpoints from which we want to render the mesh.
num_views = 10

# Get a batch of viewing angles.
elev = torch.linspace(0, 360, num_views)
azim = torch.linspace(-180, 180, num_views)

# Place a point light in front of the object. As mentioned above, the front of
# the cow is facing the -z direction.
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Initialize an OpenGL perspective camera that represents a batch of different
# viewing angles. All the cameras helper methods support mixed type inputs and
# broadcasting. So we can view the camera from the a distance of dist=2.7, and
# then specify elevation and azimuth angles for each viewpoint as tensors.
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = PerspectiveCameras(device=device, R=R, T=T)

# We arbitrarily choose one particular view that will be used to visualize
# results
camera = PerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...])


# Define the settings for rasterization and shading. Here we set the output
# image to be of size 128X128. As we are rendering images for visualization
# purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
# rasterize_meshes.py for explanations of these parameters.  We also leave
# bin_size and max_faces_per_bin to their default values of None, which sets
# their values using heuristics and ensures that the faster coarse-to-fine
# rasterization method is used.  Refer to docs/notes/renderer.md for an
# explanation of the difference between naive and coarse-to-fine rasterization.
raster_settings = RasterizationSettings(
    image_size=128, blur_radius=0.0, faces_per_pixel=1, perspective_correct=False
)

# Create a Phong renderer by composing a rasterizer and a shader. The textured
# Phong shader will interpolate the texture uv coordinates for each vertex,
# sample from a texture image and apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=camera, lights=lights),
)

# Create a batch of meshes by repeating the cow mesh and associated textures.
# Meshes has a useful `extend` method which allows us do this very easily.
# This also extends the textures.
meshes = mesh.extend(num_views)

# Render the cow mesh from each viewing angle
target_images = renderer(meshes, cameras=cameras, lights=lights)

# Our multi-view cow dataset will be represented by these 2 lists of tensors,
# each of length num_views.
target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
target_cameras = [
    PerspectiveCameras(device=device, R=R[None, i, ...], T=T[None, i, ...])
    for i in range(num_views)
]


# RGB images
image_grid(target_images.cpu().numpy(), rows=4, cols=5, rgb=True)
# plt.show()


# Rasterization settings for silhouette rendering
sigma = 1e-4
raster_settings_silhouette = RasterizationSettings(
    image_size=128,
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
    faces_per_pixel=50,
)

# Silhouette renderer
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, raster_settings=raster_settings_silhouette
    ),
    shader=SoftSilhouetteShader(),
)

# Render silhouette images.  The 3rd channel of the rendering output is
# the alpha/silhouette channel
silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]

# Visualize silhouette images
image_grid(silhouette_images.cpu().numpy(), rows=4, cols=5, rgb=False)
# plt.show()
# Show a visualization comparing the rendered predicted mesh to the ground truth
# mesh
def visualize_prediction(
    predicted_mesh,
    renderer=renderer_silhouette,
    target_image=target_rgb[1],
    title="",
    silhouette=False,
):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()


# Plot losses as a function of optimization iteration
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l["values"], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    plt.show()


# We initialize the source shape to be a sphere of radius 1.
src_mesh = ico_sphere(4, device)

# Rasterization settings for differentiable rendering, where the blur_radius
# initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable
# Renderer for Image-based 3D Reasoning', ICCV 2019
sigma = 1e-4
raster_settings_soft = RasterizationSettings(
    image_size=128,
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
    faces_per_pixel=50,
    perspective_correct=False,
)

# Silhouette renderer
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings_soft),
    shader=SoftSilhouetteShader(),
)

# Differentiable soft renderer using per vertex RGB colors for texture
renderer_textured = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings_soft),
    shader=SoftPhongShader(device=device, cameras=camera, lights=lights),
)

# Number of views to optimize over in each SGD iteration
num_views_per_iteration = 2
# Number of optimization steps
Niter = 3000
# Plot period for the losses
plot_period = 500

WEIGHTS = [1.0, 1.0, 0.01, 1.0, 1.0]

# Optimize using rendered silhouette image loss, mesh edge loss, mesh normal
# consistency, and mesh laplacian smoothing
losses = {
    "silhouette": {"weight": WEIGHTS[0], "values": []},
    "edge": {"weight": WEIGHTS[1], "values": []},
    "normal": {"weight": WEIGHTS[2], "values": []},
    "laplacian": {"weight": WEIGHTS[3], "values": []},
    "rgb": {"weight": WEIGHTS[4], "values": []},
}

# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters is equal to the total number of vertices in
# src_mesh
verts_shape = src_mesh.verts_packed().shape
deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

# We will also learn per vertex colors for our sphere mesh that define texture
# of the mesh
sphere_verts_rgb = torch.full(
    [1, verts_shape[0], 3], 0.8, device=device, requires_grad=True
)

# The optimizer
# optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
optimizer = torch.optim.SGD([deform_verts, sphere_verts_rgb], lr=1.0, momentum=0.9)

loop = tqdm(range(Niter))

for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()

    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    # Add per vertex colors to texture the mesh
    new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)
    # Losses to smooth /regularize the mesh shape

    loss = {k: torch.tensor(0.0, device=device) for k in losses}
    update_mesh_shape_prior_losses(new_src_mesh, loss)

    # silhouette optimize
    if False:
        # Compute the average silhouette loss over two random views, as the average
        # squared L2 distance between the predicted silhouette and the target
        # silhouette from our dataset
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            images_predicted = renderer_silhouette(
                new_src_mesh, cameras=target_cameras[j], lights=lights
            )
            predicted_silhouette = images_predicted[..., 3]
            loss_silhouette = (
                (predicted_silhouette - target_silhouette[j]) ** 2
            ).mean()
            loss["silhouette"] += loss_silhouette / num_views_per_iteration

        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))

    # texture and silhouette optimize
    else:
        # Randomly select two views to optimize over in this iteration.  Compared
        # to using just one view, this helps resolve ambiguities between updating
        # mesh shape vs. updating mesh texture
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            images_predicted = renderer_textured(
                new_src_mesh, cameras=target_cameras[j], lights=lights
            )

            # Squared L2 distance between the predicted silhouette and the target
            # silhouette from our dataset
            predicted_silhouette = images_predicted[..., 3]
            loss_silhouette = (
                (predicted_silhouette - target_silhouette[j]) ** 2
            ).mean()
            loss["silhouette"] += loss_silhouette / num_views_per_iteration

            # Squared L2 distance between the predicted RGB image and the target
            # image from our dataset
            predicted_rgb = images_predicted[..., :3]
            loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
            loss["rgb"] += loss_rgb / num_views_per_iteration

        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))

    # Print the losses
    loop.set_description("total_loss = %.6f" % sum_loss)

    # Plot mesh
    if False and i % plot_period == 0:
        visualize_prediction(
            new_src_mesh,
            renderer=renderer_textured,
            title="iter: %d" % i,
            silhouette=False,
            target_image=target_rgb[1],
        )

    # Optimization step
    sum_loss.backward()
    optimizer.step()


# Fetch the verts and faces of the final predicted mesh
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# Scale normalize back to the original target size
final_verts = final_verts * scale + center

# Store the predicted mesh using save_obj
final_obj = os.path.join("./morphy_results", "final_from_depth.ply")
# save_obj(final_obj, final_verts, final_faces)
IO().save_mesh(new_src_mesh, final_obj, binary=False, colors_as_uint8=True)


# visualization of 20 cameras view
new_src_meshes = new_src_mesh.extend(num_views)
result_images = renderer(new_src_meshes, cameras=cameras, lights=lights)
result_rgbs = [result_images[i, ..., :3] for i in range(num_views)]
image_grid(result_images.detach().cpu().numpy(), rows=4, cols=5, rgb=True)

# single view and losses visualziation
visualize_prediction(
    new_src_mesh,
    renderer=renderer_textured,
    silhouette=False,
    target_image=target_rgb[1],
)
plot_losses(losses)
