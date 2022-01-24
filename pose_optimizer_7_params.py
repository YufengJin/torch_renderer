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
from pytorch3d.transforms import (
    quaternion_to_matrix,
    quaternion_apply,
    matrix_to_quaternion
)
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
# track the gradient nan error
#torch.autograd.set_detect_anomaly(True)
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
silhouette_raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=50, 
    # NOTE
    perspective_correct = False
)


# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=silhouette_raster_settings
    ),
    #shader=SoftSilhouetteShader(blend_params=blend_params)
    shader=SoftSilhouetteShader()
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
    #shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# decompose Meshrendrer into two parts
rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=silhouette_raster_settings
    )

SilhouetteShader = SoftSilhouetteShader()

PhongShader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
# Select the viewpoint using spherical angles  
distance = 5.   # distance from camera to the object
elevation = 50.0
azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
print('reference camera R, t:\n', R, T)
# Render the teapot providing the values of R and T. 

# reference (T,quaternion) = [0.0000, -0.0000,  5.0000,  0.0000,  0.0000,  0.9063, -0.4226]
quaternion_ref = torch.cat((T,matrix_to_quaternion(R)), axis = -1)
print('reference quaternion: ', quaternion_ref)


#fragments = rasterizer(teapot_mesh, R=R, T=T)
#silhouette = fragments.zbuf
#image_ref = PhongShader(fragments, teapot_mesh, R=R, T=T)

silhouette = silhouette_renderer(teapot_mesh, R=R, T=T)
image = phong_renderer(teapot_mesh, R=R, T=T)
# plot the colored and silhouette images test
silhouette_ref = silhouette.cpu().numpy()
image_ref = image.cpu().numpy()


print(silhouette_ref.shape,  image_ref.shape)
if False:

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouette_ref.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(image_ref.squeeze()[..., 3])
    plt.grid(False)
    plt.show()


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref, calc_grad=False):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        # TODO  
        #image_ref = torch.from_numpy(image_ref[...,3]).to(meshes.device)
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))

        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([-0.3400, 0.3000,  8.0000,  0.5000,  -0.100,  -0.9063, -0.4226], dtype=np.float32)).to(meshes.device))

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the 
        
        # quaternion normalization 
        quat = self.camera_position[3:].clone()
        normalized_quat = quat/(((quat**2).sum())**0.5)

        print('normalized quat: ', self.camera_position)        
        R = quaternion_to_matrix(normalized_quat[None, :])
        T = self.camera_position[None, :3]

        depths = self.renderer(self.meshes.clone(), R=R, T=T)

        # TODO logical function not derivative
        loss = torch.sum((depths[...,3]-self.image_ref)**2) + 1e5*(((quat**2).sum()-1)**2) 

        print('depth loss: ', loss, '\tpos loss: ', self.camera_position.detach().cpu().numpy()-quaternion_ref.detach().cpu().numpy().squeeze()
              , 'pos loss sum: ', ((self.camera_position.detach().cpu().numpy()-quaternion_ref.detach().cpu().numpy().squeeze())**2).sum())
        return loss, depths 

    def _calc_depth_confidence(self, depths):
        inter_mask = torch.logical_and((depths[...,3] != -1).detach(), (self.image_ref != -1).detach())
        union_mask     = torch.logical_or((depths[...,3] != -1).detach(), (self.image_ref != -1).detach())
        
        #depths_np = (depths[...,3]!= -1).detach().cpu().numpy()
        #ref_np = (self.image_ref!= -1).detach().cpu().numpy()

        #print(depths_np.shape, ref_np.shape)
        #inter_mask = np.logical_and(depths_np, ref_np)
        #print(inter_mask.shape, inter_mask[0].sum())
        
        if False:
            plt.subplot(2,2,1); plt.imshow(inter_mask[0].squeeze().cpu().numpy())
            plt.subplot(2,2,2); plt.imshow(union_mask[0].squeeze().cpu().numpy())
            plt.subplot(2,2,3); plt.imshow(depths[...,3].squeeze().detach().cpu().numpy())
            plt.subplot(2,2,4); plt.imshow(self.image_ref.squeeze().detach().cpu().numpy())
            plt.show()
        if union_mask.sum() == 0.:
            iou_loss = inter_mask.sum()/1e-5
        else:
            iou_loss = inter_mask.sum()/union_mask.sum()
            print('iou loss: %d/%d,  %.6f'%(inter_mask.sum(), union_mask.sum(), iou_loss))
        depth_loss = ((depths[...,3][inter_mask] - self.image_ref[inter_mask])**2).sum()
        return 1 - iou_loss 
# Initialize a model


model = Model(meshes=teapot_mesh, renderer= silhouette_renderer, image_ref=image_ref).to(device)
# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
# visualize the starting postion and the reference position

_, image_init = model()
#plt.figure(figsize=(10, 10))
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
Maxiters = 400

loss_hist = list()
pose_hist = list()
grad_hist = list()
for i in range(Maxiters):
    print('\n------------------%d/%d------------------- '%(i, Maxiters))
    optimizer.zero_grad()
    print('camera pose before  : ', model.camera_position)
    pose_hist.append(model.camera_position.detach().cpu().numpy())
    loss, image = model()
    loss_hist.append(loss.detach().cpu().numpy()) 
    loss.backward()
    grad_hist.append(model.camera_position.grad.detach().cpu().numpy())
    print('camera pose gradient: ', model.camera_position.grad.detach().cpu().numpy())
    optimizer.step()
    print('camera pose after   : ', model.camera_position)
    if False  and (i % 20 ==0):
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
    
    #loop.set_description('Optimizing (loss %.4f)' % loss)
    
print('shape consistancy: ', len(loss_hist), len(pose_hist), len(grad_hist))


_, image = model()
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
plt.grid(False)
plt.title("start position")

plt.subplot(2, 3, 2)
plt.imshow(model.image_ref.cpu().numpy().squeeze())
plt.grid(False)
plt.title("Reference silhouette")

plt.subplot(2, 3, 3)
plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3]-model.image_ref.cpu().numpy().squeeze())
plt.grid(False)
plt.title("Start diff position")

plt.subplot(2, 3, 4)
plt.imshow(image.detach().squeeze().cpu().numpy()[..., 3])
plt.grid(False)
plt.title("Optimized position")

plt.subplot(2, 3, 5)
plt.imshow(model.image_ref.cpu().numpy().squeeze() - image.detach().squeeze().cpu().numpy()[..., 3])
plt.grid(False)
plt.title("Depth diff")


plt.subplot(2, 3, 6)
plt.plot(loss_hist)
plt.title("Depth decrepancy loss")

#plt.savefig('./results/result.png', format = 'png', dpi=200)
plt.figure(figsize=(15, 10))  
plt.subplot(1,2,1)
plt.plot(pose_hist, label=['x','y','z','qw','qx','qy','qz'])
plt.legend()
plt.title("poses")

plt.subplot(1,2,2)
plt.plot(grad_hist, label=['x','y','z','qw','qx','qy','qz'])
plt.legend()

plt.title("gradients")
plt.show()

