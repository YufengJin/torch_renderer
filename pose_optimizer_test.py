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
obj_name = '004_sugar_box'
obj_file = "/hri/localdisk2/yjin/YCB_Video_Dataset/models/%s/textured.obj"% obj_name

# Create a Meshes object. Here we have only one mesh in the batch.
mesh = load_objs_as_meshes([obj_file], device=device)

verts = mesh.verts_packed()
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)))

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
    faces_per_pixel=1, 
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


PhongShader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
# Select the viewpoint using spherical angles  
distance = 4.   # distance from camera to the object
elevation = 50.0
azimuth = 30.0  # No rotation so the camera is positioned on the +Z axis. 

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
print('reference camera R, t:\n', R, T)

# reference (T,quaternion) = [0.0000, -0.0000,  5.0000,  0.0000,  0.0000,  0.9063, -0.4226]
quaternion_ref = torch.cat((T,matrix_to_quaternion(R)), axis = -1)
print('reference quaternion: ', quaternion_ref)

fragments = rasterizer(mesh, R=R, T=T)

silhouette = fragments.zbuf.mean(-1)
image_ref = phong_renderer(mesh, R=R, T=T)

# plot the colored and silhouette images test
silhouette_ref = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()

# refill the background with 0
silhouette_ref[silhouette_ref == -1] = 0.

if False:

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouette_ref.squeeze())  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(image_ref.squeeze()[..., :3])
    plt.grid(False)
    plt.show()


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # delta for calculate gradient of quaternion and tranlaion 
        delta = np.repeat([0.01, 0.01], [4,4])
      
        # 8 by 7 matrix with zeros on first rows
        delta_mat = np.diag(delta)[:, 1:]

        self.delta_mat = torch.from_numpy(delta_mat.astype(np.float32)).to(self.device)
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        # TODO render true depth 
        image_ref = torch.from_numpy(image_ref).to(meshes.device)

        self.register_buffer('images_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        #self.camera_position   = nn.Parameter(torch.from_numpy(np.array([0.0400, 0.010,  3.7000,  -0.1,  0.1, .9063, -0.426], dtype=np.float32)).to(meshes.device))
        self.camera_position   = nn.Parameter(torch.from_numpy(np.array([0.0400, 0.010,  3.7000], dtype=np.float32)).to(meshes.device))
        self.camera_quaternion = nn.Parameter(torch.from_numpy(np.array([-0.1,  0.1, .9063, -0.426], dtype=np.float32)).to(meshes.device))

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the 
        
        # quaternion normalization 
        quat = self.camera_quaternion.clone()
        #quat = self.camera_rotation.clone()
        normalized_quat = quat/(((quat**2).sum())**0.5)

        R = quaternion_to_matrix(normalized_quat[None, :])
        T = self.camera_position[None,:]

        #elevation, azimuth, distance = self.camera_position
        #R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

        #print('rendering 1 meshes, checking shape of R,t, and lens of meshes', R.shape, T.shape,len(self.meshes))
        fragments = self.renderer(self.meshes.clone(), R=R, T=T)
        depths = fragments.zbuf.mean(-1)
        # NOTE regularization 1e5 before
        loss = self._calc_depth_confidence(depths)[0] #+ 1*(((quat**2).sum()-1)**2) 
         
        return loss, depths 

    def calc_gradient(self):
        with torch.no_grad():
            quat = self.camera_quaternion.clone()
            normalized_quat = quat/(((quat**2).sum())**0.5)

            camera_pos = torch.zeros(7).to(self.device)
            camera_pos[:3] = self.camera_position.clone()
            camera_pos[3:] = normalized_quat

            cameras_mat = camera_pos[None,:].repeat(8, 1) + self.delta_mat
            
            #print("DEBUG: cameras_mat: ", cameras_mat)
            R = quaternion_to_matrix(cameras_mat[...,3:])
            T = cameras_mat[...,:3] 
      
            
            fragments = self.renderer(self.meshes.extend(8), R=R, T=T)
     
            depths = fragments.zbuf.mean(-1)
            #print("DEBUG: depths shape: ", depths.shape)
            _ , losses = self._calc_depth_confidence(depths)
            
            delta_losses = losses[1:] - losses[0]
            #print("origin loss: ", losses)
            #print('delta losses: ', delta_losses)
            return delta_losses.cpu().numpy() 

    def _calc_depth_confidence2(self, depths, amp = 1e5, clip_tol = 0.05):
        # mask intersection image
        #TODO  sigmoid and tahn? simoid(0) = 0.5, tanh(0)=0
        #inter_mask = torch.sigmoid(amp*depths*self.images_ref)

        if depths.shape[0] != 1:
            images_ref = self.images_ref.repeat(depths.shape[0],1,1)
        else:
            images_ref = self.images_ref.clone()
           

        inter_mask = torch.tanh(amp*(depths+1.)*(images_ref))

        #union_mask = torch.sigmoid(amp*(depths+self.images_ref))
        union_mask = torch.tanh(amp*(depths+images_ref+1.))
        

        # TODO if there is no intersection mask
        iou = inter_mask.sum((1,2))/union_mask.sum((1,2))
        dist_clip  =  torch.clip(torch.abs(depths - images_ref), 0., clip_tol) 

        dist_score = iou*(1-(dist_clip*inter_mask/clip_tol).mean((1,2)))
        print('inter union mask shape: ', inter_mask.shape, union_mask.shape, '\tiou = {},  dist= {}'.format(iou, dist_score))

        
        if False and depths.shape[0] == 1:
            plt.figure(figsize=(12, 10))
            plt.subplot(2,3,1); plt.imshow(inter_mask[0].detach().cpu().numpy()); plt.title("intersect")
            plt.subplot(2,3,2); plt.imshow(union_mask[0].detach().cpu().numpy()); plt.title("union")
            plt.subplot(2,3,3); plt.imshow(depths[0].detach().cpu().numpy()); plt.title('rendered depth'); plt.colorbar()
            plt.subplot(2,3,4); plt.imshow(images_ref[0].detach().cpu().numpy()) ; plt.title('gt depth'); plt.colorbar()
            plt.subplot(2,3,5); plt.imshow((depths-images_ref)[0].detach().cpu().numpy()); plt.title('Diff depth'); plt.colorbar()
            plt.subplot(2,3,6); plt.imshow((dist_clip*inter_mask)[0].detach().cpu().numpy()); plt.title('intersect depth'); plt.colorbar()
            plt.suptitle("depth confidence")
            plt.show()


        #print("DEBUG: iou, ", iou, " , dist_core, ", dist_score)
        return (1-iou).sum(), (1-iou)

    def _calc_depth_confidence(self, depths, amp = 1e5, clip_tol = 0.05):
        # mask intersection image
        #TODO  sigmoid and tahn? simoid(0) = 0.5, tanh(0)=0
        #inter_mask = torch.sigmoid(amp*depths*self.images_ref)

        if depths.shape[0] != 1:
            images_ref = self.images_ref.repeat(depths.shape[0],1,1)
        else:
            images_ref = self.images_ref.clone()
           

        inter_mask = torch.tanh(amp*(depths+1.)*(images_ref))

        #union_mask = torch.sigmoid(amp*(depths+self.images_ref))
        union_mask = torch.tanh(amp*(depths+images_ref+1.))
        

        # TODO if there is no intersection mask
        iou = inter_mask.sum((1,2))/union_mask.sum((1,2))
        dist_clip  =  torch.clip(torch.abs(depths - images_ref), 0., clip_tol) 

        dist_score = iou*(1-(dist_clip*inter_mask/clip_tol).mean((1,2)))
        #print('inter union mask shape: ', inter_mask.shape, union_mask.shape, '\tiou = {},  dist= {}'.format(iou, dist_score))

        
        if False and depths.shape[0] == 1:
            plt.figure(figsize=(12, 10))
            plt.subplot(2,3,1); plt.imshow(inter_mask[0].detach().cpu().numpy()); plt.title("intersect")
            plt.subplot(2,3,2); plt.imshow(union_mask[0].detach().cpu().numpy()); plt.title("union")
            plt.subplot(2,3,3); plt.imshow(depths[0].detach().cpu().numpy()); plt.title('rendered depth'); plt.colorbar()
            plt.subplot(2,3,4); plt.imshow(images_ref[0].detach().cpu().numpy()) ; plt.title('gt depth'); plt.colorbar()
            plt.subplot(2,3,5); plt.imshow((depths-images_ref)[0].detach().cpu().numpy()); plt.title('Diff depth'); plt.colorbar()
            plt.subplot(2,3,6); plt.imshow((dist_clip*inter_mask)[0].detach().cpu().numpy()); plt.title('intersect depth'); plt.colorbar()
            plt.suptitle("depth confidence")
            plt.show()


        #print("DEBUG: iou, ", iou, " , dist_core, ", dist_score)
        return (1-(iou+dist_score)/2).sum(), 1-(iou+dist_score)/2
# Initialize a model


model = Model(meshes=mesh, renderer= rasterizer, image_ref=silhouette_ref).to(device)
# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

# visualize the starting postion and the reference position

_, image_init = model()


if False:
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.detach().squeeze().cpu().numpy())
    plt.grid(False)
    plt.title("Starting position")
    
    plt.subplot(1, 2, 2)
    plt.imshow(model.image_ref.cpu().numpy().squeeze())
    plt.grid(False)
    plt.title("Reference silhouette")
    plt.show()



# optimiztion run
Maxiters = 200

loss_hist = list()
pose_hist = list()
grad_hist = list()

def cosine_distance(x,y):
    norm = (x*y).sum()
    dorm = np.sqrt((x**2).sum())*np.sqrt((y**2).sum())
    return norm/dorm
    
for i in range(Maxiters):
    print('\n------------------%d/%d------------------- '%(i, Maxiters))
    optimizer.zero_grad()
    #optimizer2.zero_grad()
    #print('>>> camera pose before  : ', model.camera_position
    pose_hist.append(np.concatenate([model.camera_position.detach().cpu().numpy(), model.camera_quaternion.detach().cpu().numpy()]))

    loss, image = model()
    loss_hist.append(loss.detach().cpu().numpy()) 

    print("%d/%d loss: %.6f"%(i, Maxiters, loss.detach().cpu().numpy()))
    loss.backward()

    grad_hist.append(np.concatenate([model.camera_position.grad.detach().cpu().numpy(), model.camera_quaternion.grad.detach().cpu().numpy()]))

    print('>>> camera pose cos distance: ', cosine_distance(model.camera_position.grad.detach().cpu().numpy(),  model.calc_gradient()[:3]))
    print('>>> camera rot cos distance: ', cosine_distance(model.camera_quaternion.grad.detach().cpu().numpy(), model.calc_gradient()[3:]))
    print('>>> camera all cos distance: ', cosine_distance(np.concatenate([model.camera_position.grad.detach().cpu().numpy(), model.camera_quaternion.grad.detach().cpu().numpy()]), model.calc_gradient()))
    optimizer.step()
    #optimizer2.step()
    #print('>>> camera pose after   : ', model.camera_position)
     
    if True and (i % 5 ==0):
        plt.clf() 
        plt.subplot(2, 3, 1)
        plt.imshow(image.detach().squeeze().cpu().numpy())
        plt.grid(False)
        plt.title("render silhouette")
        
        plt.subplot(2, 3, 2)
        plt.imshow(model.images_ref.cpu().numpy().squeeze())
        plt.grid(False)
        plt.title("Reference silhouette")

        plt.subplot(2, 3, 3)
        plt.imshow(model.images_ref.cpu().numpy().squeeze() - image.detach().squeeze().cpu().numpy())
        plt.grid(False)
        plt.title("Diff silhouette")
        plt.colorbar()

        plt.subplot(2, 3, 4)
        plt.plot(loss_hist)
        plt.ylim((0,0.5))
        plt.title("Loss")

        plt.subplot(2,3,5)
        plt.plot(pose_hist, label=['x','y','z','qw','qx','qy','qz'])
        #plt.legend()
        plt.title("Poses")
        
        plt.subplot(2,3,6)
        plt.plot(grad_hist, label=['x','y','z','qw','qx','qy','qz'])
        #plt.legend()
        plt.title("Gradients")

        plt.pause(0.01)
        #plt.show()
        #plt.savefig('./results/iter_%d.png'%i, format = 'png', dpi=200)
    
    #loop.set_description('Optimizing (loss %.4f)' % loss)
    
_, image = model()
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_init.detach().squeeze().cpu().numpy())
plt.grid(False)
plt.title("start position")

plt.subplot(2, 3, 2)
plt.imshow(model.images_ref.cpu().numpy().squeeze())
plt.grid(False)
plt.title("Reference silhouette")

plt.subplot(2, 3, 3)
plt.imshow(image_init.detach().squeeze().cpu().numpy()-model.images_ref.cpu().numpy().squeeze())
plt.grid(False)
plt.title("Start diff position")

plt.subplot(2, 3, 4)
plt.imshow(image.detach().squeeze().cpu().numpy())
plt.grid(False)
plt.title("Optimized position")

plt.subplot(2, 3, 5)
plt.imshow(model.images_ref.cpu().numpy().squeeze() - image.detach().squeeze().cpu().numpy())
plt.grid(False)
plt.title("Depth diff")


plt.subplot(2, 3, 6)
plt.plot(loss_hist)
plt.title("Depth decrepancy loss")

#plt.savefig('./results/result.png', format = 'png', dpi=200)
if False:
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
