import os
import sys
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm
import time
import torch.nn as nn
import torch.nn.functional as F
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
# conversion from opencv to pytorch3d
from pytorch3d.utils import cameras_from_opencv_projection
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
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


# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj("./data/teapot.obj")
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
teapot_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)


# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=50, 
    perspective_correct = False
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# compose Silhouette rendering into

# Select the viewpoint using spherical angles  
distance = 5.   # distance from camera to the object
elevation = 50.0
azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
#print('reference camera R, t:\n', R, T)
# Render the teapot providing the values of R and T. 
silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)

# plot the colored and silhouette images test
silhouette_ref = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()

class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        #image_ref = torch.from_numpy(image_ref[...,3]).to(meshes.device)

        # Get depth mask
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([3.0,  6.9, 2.5], dtype=np.float32)).to(meshes.device))

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        #R, T = look_at_view_transform(self.camera_position[0][None],self.camera_position[1][None], self.camera_position[2][None],  device=device)

        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

        
        loss = torch.sum((image[..., 3] - self.image_ref)**2)
        return loss, image


# Initialize a model
model = Model(meshes=teapot_mesh, renderer=silhouette_renderer, image_ref=image_ref).to(device)


#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

_, image_init = model()
# visualize the starting postion and the reference position
#plt.figure(figsize=(10, 10))
#
#plt.subplot(1, 2, 1)
#plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
#plt.grid(False)
#plt.title("Starting position")
#
#plt.subplot(1, 2, 2)
#plt.imshow(model.image_ref.cpu().numpy().squeeze())
#plt.grid(False)
#plt.title("Reference silhouette")
#plt.show()



# optimiztion run
MaxIters = 1000

loss_hist = list()
for i in range(MaxIters):
    tic1 = time.time()
    optimizer.zero_grad()
    loss, image = model()
    tic2 = time.time()
    loss_hist.append(loss.detach().cpu().numpy()) 
    loss.backward()
    tic3 = time.time()
    print('%d/%d   loss       : %.4f'%(i, MaxIters, loss))
    print('%d/%d   camera pos : '%(i, MaxIters) , model.camera_position)
    print('%d/%d   camera grad: '%(i, MaxIters) , model.camera_position.grad)

    optimizer.step()
    print('forward time: ', tic2-tic1, '\tloss backward time: ', tic3-tic2, '\tstep back time: ', time.time()-tic3)
    if False and (i % 20 ==0):
        plt.figure(figsize=(10, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image.detach().squeeze().cpu().numpy()[..., 3])
        plt.grid(False)
        plt.title("current position")
        
        plt.subplot(1, 2, 2)
        plt.imshow(model.image_ref.cpu().numpy().squeeze())
        plt.grid(False)
        plt.title("Reference silhouette")
        plt.suptitle('iters: %d'%i)
        plt.show()
        #plt.savefig('./results/iter_%d.png'%i, format = 'png', dpi=200)
    
    
_, image = model()
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
plt.grid(False)
plt.title("Start silhouette")


plt.subplot(2, 3, 2)
plt.imshow(model.image_ref.cpu().numpy().squeeze())
plt.grid(False)
plt.title("Reference silhouette")


plt.subplot(2, 3, 4)
plt.imshow(image.detach().squeeze().cpu().numpy()[..., 3])
plt.grid(False)
plt.title("Optimized silhouette")

plt.subplot(2, 3, 5)
plt.imshow(model.image_ref.cpu().numpy().squeeze()-image.detach().squeeze().cpu().numpy()[..., 3])
plt.grid(False)
plt.title("Discrepancy")

plt.subplot(2, 3, 3)
plt.plot(loss_hist)
plt.title("Loss")
#plt.savefig('./results/result.png', format = 'png', dpi=200)
plt.show()

