import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import time
import torch.nn as nn
import torch.nn.functional as F
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
# conversion from opencv to pytorch3d
from pytorch3d.utils import cameras_from_opencv_projection
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (
    quaternion_to_matrix,
    quaternion_apply,
    matrix_to_quaternion
)
from collections import defaultdict

from pytorch3d.renderer import (
    look_at_rotation,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,

)

# colormapping
colormap = plt.cm.viridis 
norm = lambda X: (X-X.min())/(X.max()-X.min())

PATCH_SIZE = (200, 200)  # Example patch size

import wandb
# Initialize wandb with your project name
wandb.init(project="depth-optimization")


torch.autograd.set_detect_anomaly(True)

def patch_image(image):
    #np.random.seed(50)
    random_patch_mask = np.zeros_like(image, dtype=np.uint8)
    # find non-zero depth
    X, Y, Z = np.where(image != 0.)

    idx = np.random.randint(0, len(X)//4)
    start_x, start_y = X[idx], Y[idx]

    end_x = start_x + PATCH_SIZE[1]
    end_y = start_y + PATCH_SIZE[0]
    random_patch_mask[start_y:end_y, start_x:end_x] = 255

    masked_image = np.copy(image)
    masked_image[random_patch_mask == 0] = 0.

    return masked_image



# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Load the obj and ignore the textures and materials.
object_p = "data/cow_mesh/cow.obj" 
verts, faces_idx, _ = load_obj(object_p)
#erts, faces_idx, _ = load_obj("./data/teapot.obj")
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
meshes = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

meshes = load_objs_as_meshes([object_p], device=device)

# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color= (0, 0, 0))

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
#raster_settings = RasterizationSettings(
#    image_size=512, 
#    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
#    faces_per_pixel=50, 
#    perspective_correct = False
#)

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0., 
    faces_per_pixel=1, 
    #perspective_correct = False
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

fragments = rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# SoftPhongShader
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params
    )
)


# Select the viewpoint using spherical angles  
distance = 0.7   # distance from camera to the object
elevation = 30.0   # angle of elevation in degrees
azimuth = 60.0  # No rotation so the camera is positioned on the +Z axis. 

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

# reference (T,quaternion) = [0.0000, -0.0000,  5.0000,  0.0000,  0.0000,  0.9063, -0.4226]
quaternion_ref = torch.cat((T, matrix_to_quaternion(R)), axis = -1)
print('reference quaternion: ', quaternion_ref, quaternion_ref.size())
quaternion_ref = quaternion_ref.cpu().numpy()

# Render the teapot providing the values of R and T. 
silhouette_ref = silhouette_renderer(meshes_world=meshes, R=R, T=T)
depth_ref = fragments(meshes_world=meshes, R=R, T=T).zbuf
rgb_ref = renderer(meshes_world=meshes, R=R, T=T)[..., :3]

silhouette_ref = silhouette_ref.cpu().numpy()
silhouette_ref = (silhouette_ref[...,3] != 0.).astype(np.uint8)

depth_ref = depth_ref[..., 0].cpu().numpy()
depth_ref[depth_ref == -1.] = 0.
rgb_full_ref = rgb_ref.cpu().numpy()

# Patch the image with the mask
image = depth_ref.transpose(1,2,0).copy()
depth_ref = patch_image(image).transpose(2, 0, 1)

silhouette_ref = (depth_ref != 0.)

rgb_ref = np.zeros_like(rgb_full_ref)
rgb_ref[silhouette_ref] = rgb_full_ref[silhouette_ref]

plt.subplot(1, 3, 1); plt.imshow(silhouette_ref.transpose(1,2,0))
plt.subplot(1, 3, 2); plt.imshow(depth_ref.transpose(1,2,0))
plt.subplot(1, 3, 3); plt.imshow(rgb_ref[0])
plt.show()


mseloss = torch.nn.MSELoss()
huberloss = torch.nn.HuberLoss(delta=0.05)
l1loss = torch.nn.L1Loss()

class Model(nn.Module):
    def __init__(self, meshes, rasterizer, silhouette_renderer, phong_render, depth_ref, silhouette_ref, rgb_ref, cam_pose_gt):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.rasterizer = rasterizer
        self.silhouette_renderer = silhouette_renderer
        self.phong_render = phong_render
        
        self.losses = defaultdict(list) 
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        depth_ref = torch.from_numpy(depth_ref).float()
        silhouette_ref = torch.from_numpy(silhouette_ref).bool()
        rgb_ref = torch.from_numpy(rgb_ref).float()

        self.register_buffer('depth_ref', depth_ref)
        self.register_buffer('silhouette_ref', silhouette_ref)
        self.register_buffer('rgb_ref', rgb_ref)

        # camera pose gt 
        self._cam_pose_gt = cam_pose_gt
        initPose = cam_pose_gt
        
        #np.random.seed(100)
        # add error on pose
        err = np.random.randn(1,7) * 0.01 * 3 
        initPose += err 
 
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position = nn.Parameter(torch.from_numpy(initPose).float().to(meshes.device))
        #self.camera_position   = nn.Parameter(torch.from_numpy(np.array([0.000, 0.0,  0.7000,  1.,  0., .9, -0.4], dtype=np.float32)).to(meshes.device))

    def forward(self):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        R = quaternion_to_matrix(self.camera_position[:, 3:]).to(self.device)
        T = self.camera_position[:,:3]
        
        fragments = self.rasterizer(meshes_world=self.meshes, R=R, T=T)
        depth_zbuf = fragments.zbuf[...,0]
        depth = torch.relu(depth_zbuf)

        silhouette = self.silhouette_renderer(self.meshes, R=R, T=T)[..., 3]

        color = self.phong_render(self.meshes, R=R, T=T)[..., :3]

        # Calculate the silhouette loss
        loss = self.calc_loss(depth, silhouette , color) 
        return loss, depth, color

    #TODO change to depth reason negative evidence
    def calc_loss(self, depth, depth_mask, color):
        mask = self.silhouette_ref

        sil_loss = l1loss(depth_mask, mask.float())

        color_loss = mseloss(color, self.rgb_ref)

        depth_gt = torch.masked_select(self.depth_ref, mask)
        depth = torch.masked_select(depth, mask)

        depthLoss = mseloss(depth, depth_gt) 

        hloss = huberloss(depth, depth_gt)
        wandb.log({"depthMSE": depthLoss})
        wandb.log({"colorMSE": color_loss})
        wandb.log({"silhouette_l1": sil_loss})
        wandb.log({"depthHuber": hloss})


        return sil_loss+hloss+color_loss*0.01


# Initialize a model using the renderer, mesh and reference image
model = Model(meshes=meshes, rasterizer=fragments, silhouette_renderer = silhouette_renderer, phong_render = renderer, depth_ref=depth_ref, silhouette_ref=silhouette_ref, rgb_ref = rgb_ref, cam_pose_gt=quaternion_ref).to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


_, depth_init, color = model()

depth_gt = model.depth_ref.detach().squeeze().cpu().numpy()
color_gt = model.rgb_ref.detach().squeeze().cpu().numpy()
depth_init = depth_init.detach().squeeze().cpu().numpy()
diff = (depth_init-depth_gt)**2
images = np.hstack((depth_gt, depth_init, diff))
colors = colormap(norm(images))

images = wandb.Image(colors, caption="Left: target depth, Middle: initial depth, Right: depth diff")

wandb.log({"Initial Error": images})        

loop = tqdm(range(500))
for i in loop:
    optimizer.zero_grad()
    loss, depth, color = model()
    loss.backward()
    
    optimizer.step()
    
    msg = f"DEBUG: Loss: {loss.data:.4f}, \
            \nQuat losee:  {model._cam_pose_gt - model.camera_position.data.detach().cpu().numpy()}"

    loop.set_description(msg)
    
    # Save outputs to create a GIF. 
    if i % 10 == 0:
        depth = depth.detach().squeeze().cpu().numpy()
        diff = (depth-depth_gt)**2
        images = np.hstack((depth_gt, depth, diff))
        images = cv2.cvtColor(colormap(norm(images)).astype(np.float32), cv2.COLOR_BGR2RGB)
        images_color = np.ones_like(images)
        images_color[:, :512, :] = color_gt
        images_color[:,512:2*512, :] = color.detach().squeeze().cpu().numpy()
        images_color[:,2*512:, :] = norm(color.detach().squeeze().cpu().numpy() - color_gt)
        images = np.vstack((images, images_color))

        cv2.imshow("Diff", images)

        # Check for key press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break

if not isinstance(depth, np.ndarray):
    depth = depth.detach().squeeze().cpu().numpy()

diff = (depth-depth_gt)**2
images = np.hstack((depth_gt, depth, diff))
 
colors = colormap(norm(images))
images = wandb.Image(colors, caption="Left: target depth, Middle: final depth, Right: depth diff")
# Convert grayscale image to a color heatmap

wandb.log({"End Error": images})        
