import os
import sys
import torch
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
        raster_settings=raster_settings,
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
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)


rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings)


shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

# Select the viewpoint using spherical angles  
distance = 5.   # distance from camera to the object
elevation = 50.0
azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
print('reference camera R, t:\n', R, T)


# Render the teapot providing the values of R and T. 
fragments = rasterizer(teapot_mesh, R=R, T=T)
silhouette = fragments.zbuf.mean(-1)
image_ref = shader(fragments, teapot_mesh, R=R, T=T)

# plot the colored and silhouette images test
silhouette_ref = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()

print(silhouette_ref.shape, image_ref.shape)

if False:

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouette_ref.squeeze()[...])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(image_ref.squeeze())
    plt.grid(False)
    plt.show()


# generate inital R, t 
init_dist, init_elev, init_azim = 6., 50.0,-20.
#init_cam_pos = torch.from_numpy(np.array([3.0,  50., 0.], dtype=np.float32)).to(device)
init_R, init_t= look_at_view_transform(init_dist, init_elev, init_azim, device=device)
#init_t = -torch.bmm(R.transpose(1, 2), init_cam_pos[None, :, None])[:, :, 0]   # (1, 3)

print('initial camera R, t :\n', init_R, init_t)
init_R = init_R.squeeze().cpu().numpy().astype(np.float32)
init_t = init_t.squeeze().cpu().numpy().astype(np.float32)

class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref, init_R, init_t):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        # TODO loss depth info, 
        image_ref = torch.from_numpy(image_ref).to(meshes.device)
        #image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        #self.camera_position = nn.Parameter(torch.from_numpy(np.array([3.0,  6.9, +2.5], dtype=np.float32)).to(meshes.device))
        camera_pos = np.zeros((3,4), dtype=np.float32)
        camera_pos[:,:3] = init_R
        camera_pos[:, 3] = init_t
        self.camera_pos = nn.Parameter(torch.from_numpy(camera_pos).to(meshes.device))
        print('initialize the camera parameters: \n', self.camera_pos)

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        #R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        #T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        
        R = self.camera_pos[None, :, :3]
        T = self.camera_pos[None, :, 3]

        fragments = self.renderer(self.meshes.clone(), R=R, T=T)
        depths = fragments.zbuf.mean(-1)
        # Calculate the silhouette loss, add penalyty on rotation matrix
        rot_loss = ((torch.matmul(R.transpose(1,2).to(self.device), R).squeeze() - torch.eye(3).to(self.device))**2).sum().to(self.device)

        print(depths.shape, self._calc_depth_confidence(depths))
        
        loss = self._calc_depth_confidence(depths) + 1e5*rot_loss

        print('loss: ', loss, '\trot loss: ', rot_loss)
        return loss, depths 

    def _calc_depth_confidence(self, depths, amp = 1e5, clip_tol = 0.05):
        # mask intersection image
        #TODO  sigmoid and tahn? simoid(0) = 0.5, tanh(0)=0
        #inter_mask = torch.sigmoid(amp*depths*self.image_ref)
        inter_mask = torch.tanh(amp*(depths+1.)*(self.image_ref+1.))

        #union_mask = torch.sigmoid(amp*(depths+self.image_ref))
        union_mask = torch.tanh(amp*(depths+self.image_ref+2.))

        if False:
            plt.subplot(2,2,1); plt.imshow(inter_mask[0].detach().cpu().numpy())
            plt.subplot(2,2,2); plt.imshow(union_mask[0].detach().cpu().numpy())
            plt.subplot(2,2,3); plt.imshow(depths[0].detach().cpu().numpy())
            plt.subplot(2,2,4); plt.imshow(self.image_ref[0].detach().cpu().numpy())
            plt.show()

        if inter_mask.sum() == 0.:
            iou = 0
        else:
            iou = inter_mask.sum()/union_mask.sum()
        dist_clip  =  torch.clip(torch.abs(depths - self.image_ref), 0., clip_tol)
        dist_score = iou*(1-(dist_clip*inter_mask/clip_tol).mean())
        print('confidence, iou = %.4f,  dist=%.4f'%(iou, dist_score))

        return 1-(iou+dist_score)/2
# Initialize a model


# Initialize a model


model = Model(meshes=teapot_mesh, renderer=rasterizer, image_ref=silhouette_ref, init_R = init_R, init_t = init_t).to(device)
# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# visualize the starting postion and the reference position
#plt.figure(figsize=(10, 10))
#
#_, image_init = model()
#plt.subplot(1, 2, 1)
#plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
#plt.grid(False)
#plt.title("Starting position")

#plt.subplot(1, 2, 2)
#plt.imshow(model.image_ref.cpu().numpy().squeeze())
#plt.grid(False)
#plt.title("Reference silhouette")
#plt.show()



# optimiztion run
TOTAL = 400

loss_hist = list()
for i in range(TOTAL):
    print('\n%d/%d, '%(i, TOTAL))
    optimizer.zero_grad()
    loss, image = model()
    loss_hist.append(loss.detach().cpu().numpy()) 
    loss.backward()
    print('camera pos: ', model.camera_pos, '\ncamera pos grad: ',model.camera_pos.grad)

    optimizer.step()

    if False and (i % 20 ==0):
        plt.figure(figsize=(10, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image.detach().squeeze().cpu().numpy())
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
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image.detach().squeeze().cpu().numpy())
plt.grid(False)
plt.title("optimized position")

plt.subplot(1, 3, 2)
plt.imshow(model.image_ref.cpu().numpy().squeeze())
plt.grid(False)
plt.title("Reference silhouette")

plt.subplot(1, 3, 3)
plt.plot(loss_hist)
plt.title("depth decrepancy loss")
#plt.savefig('./results/result.png', format = 'png', dpi=200)
plt.show()

