import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from scipy.spatial.transform import Rotation 
# conversion from opencv to pytorch3d
from pytorch3d.utils import cameras_from_opencv_projection
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
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
    TexturesUV,
    TexturesVertex,
    BlendParams,

)
from pytorch3d.transforms import (
    quaternion_to_matrix,
    quaternion_apply,
    matrix_to_quaternion
)


import time
def camera_pose_from_opencv_to_pytorch(R,T):
    # TODO boardcasting the R,T
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1
    return R_pytorch3d, T_pytorch3d
    
def cameras_from_opencv_projection_no_ndc(R, tvec, camera_matrix, image_size, device):
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]
    print('focal_length, principal point shape: ', focal_length.shape, principal_point.shape)
    # # Retype the image_size correctly and flip to width, height.
    # image_size_wh = image_size.to(R).flip(dims=(1,))

    # # Get the PyTorch3D focal length and principal point.
    # focal_pytorch3d = focal_length / (0.5 * image_size_wh)
    # p0_pytorch3d = -(principal_point / (0.5 * image_size_wh) - 1)

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1
    return PerspectiveCameras(focal_length=focal_length,
                              principal_point=principal_point,
                              R=R_pytorch3d,
                              T=T_pytorch3d,
                              device=device,
                              in_ndc=False,
                              image_size=image_size), R_pytorch3d, T_pytorch3d

def from_tf_to_matrix(trans, rot):
    trans, rot = np.asarray(trans), np.asarray(rot)
    result = np.eye(4, dtype='float32')
    R = Rotation.from_quat(rot).as_matrix()
    result[:3, :3] = R
    result[:3, 3]  = trans
    return result


#class PoseOptimizer(nn.Module):
#    def __init__(self, meshes, renderer, image_ref, cam_init, device):
#        super().__init__()
#        self.meshes = meshes
#        self.renderer = renderer
#        self.register_buffer('image_ref', torch.from_numpy(image_ref[...,3]).type(torch.float32))
#        self.cam_mat = nn.Parameter(torch.from_numpy(cam_init).type(torch.float32).to(device))
#
#
#    def forward(self):
#        R = self.cam_mat[:,:3,:3]
#        T = self.cam_mat[:,:3,3]
          
# compare to rviz
K_cam11 =  np.array([917.700439453125, 0.0, 646.2421875, 0.0, 0.0, 915.8268432617188, 367.7502136230469, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.], dtype='float32').reshape(4,4)
K_cam12 =  np.array([914.483154296875, 0.0, 645.4754638671875, 0.0, 0., 913.0062866210938, 367.93243408203125, 0.0, 0.0, 0.,0., 1.0, 0.0, 0.0, 1.0, 0.], dtype='float32').reshape(4,4)
ext_cam11 = from_tf_to_matrix([0.008, 0.081, 0.662],[0.813, -0.270, 0.065, 0.511]) 
ext_cam12 = from_tf_to_matrix([0.233, -0.089, 0.837],[-0.058, 0.869, -0.483, -0.089])

object_trans = np.array([-4.1235436396921584e-05, -0.012795715280304881, 0.08311866360731425], dtype='float32')
object_quat  = np.array([-0.01184704775361205, -0.007628367207312281, 0.15889129486561074, 0.9871955287019907], dtype='float32')
object_mat   = from_tf_to_matrix(object_trans, object_quat)


# load from pkl files

with open('filtered_datas.pkl', 'rb') as handle:
    datas = pickle.load(handle)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


DataType = torch.float32
data = datas[1]
object_id_string = data['object_id']
extrinsic = data['extrinsic']
intrinsic = data['intrinsic']
fx, fy, cx, cy = intrinsic[0][0],intrinsic[1][1],intrinsic[0][2],intrinsic[1][2]

focal, principal = np.array([fx,fy], dtype = 'float64'), np.array([cx,cy], dtype='float64')
rendered_depth_image = data['rendered_depth']
object_mat = data['object_pose']
DATA_DIR = "/hri/storage/user/yjin/intro_ros_ws/src/intro_object_models/models/ycb-video"
obj_filename = os.path.join(DATA_DIR, "%03d/textured_simple.obj"%object_id_string)

cams = {'11':{'K': K_cam11, 'Ext': extrinsic}, '12':{'K': K_cam12, 'Ext': ext_cam12}}
#cams = {'11':{'K': K_cam11, 'Ext': ext_cam11}, '12':{'K': K_cam12, 'Ext': ext_cam12}}
image_size = (720//4,1280//4)
#image_size = (720,1280)

meshes = load_objs_as_meshes([obj_filename], device=device)



N = 10

tic_start = time.time()

#for cam in ['11', '12']:
cam = '11'
K = torch.from_numpy(cams[cam]['K']).unsqueeze(0).to(device)
#print(f'cam{cam} K with shape {K.shape}\n{K}')
K = K//4

K_3 = K[:,:3,:3].to(device)
K_3[:,2,2] = 1.
#print('K3', K_3)
camera_pose = np.dot(cams[cam]['Ext'], object_mat) 
print('translation and quaternion in opencv: ',  camera_pose[:3,3], Rotation.from_matrix(camera_pose[:3,:3]).as_quat())

# test quaternion function between pytorch and scipy
#print('numpy camera pose: ', camera_pose[:3,:3], '\n scipy matrix to quat: ', Rotation.from_matrix(camera_pose[:3,:3]).as_quat() )
camera_pose = torch.from_numpy(camera_pose).to(device) 
#print('tensor camera pose: ', camera_pose[:3,:3], '\n tensor matrix to quat: ', matrix_to_quaternion(camera_pose[:3,:3]) )
#print(quaternion_to_matrix(matrix_to_quaternion(camera_pose[:3,:3])))

R = camera_pose[:3, :3][None]
T = camera_pose[:3, 3][None]
#print(f'cam{cam} R,T with shape {R.shape} and {T.shape} \n{R}\n{T}')
# only one camera array
raster_settings = RasterizationSettings(
    image_size=image_size, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

blend_params = BlendParams(sigma = 1e-4, gamma = 1e-4)

silhouette_raster_settings = RasterizationSettings(
    image_size=image_size, 
    blur_radius=np.log(1./1e-4 -1.) * blend_params.sigma,
    # TODO figure out he meaning of arguments
    faces_per_pixel=50, 
)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

tic1 = time.time()

# extend R,T,K,Image
R = R.repeat(N,1,1); T = T.repeat(N,1); K_3 = K_3.repeat(N,1,1)    
meshes = meshes.extend(N) 
cameras, R_n, T_n = cameras_from_opencv_projection_no_ndc(R = R, tvec=T, camera_matrix = K_3, image_size = torch.tensor([image_size]), device=device)  
#cameras = PerspectiveCameras(in_ndc = False, device=device, R=R, T=T, K = K, image_size = torch.tensor([image_size]))


print('translation and quaternion in torch3d: ',  T_n.squeeze().cpu().numpy(), matrix_to_quaternion(R_n).squeeze().cpu().numpy())
print('Diff torch quat conversion and scipy , in the sequence of torch and scipy: ', matrix_to_quaternion(R_n).squeeze().cpu().numpy(), Rotation.from_matrix(R_n.squeeze().cpu().numpy()).as_quat())
tic2 = time.time()

focal_length = torch.stack([K_3[:, 0, 0], K_3[:, 1, 1]], dim=-1)
principal_point = K_3[:, :2, 2]

# intialize the cameras ionly with focal length and principal point
cameras = PerspectiveCameras(focal_length=focal_length,
                          principal_point=principal_point,
                          device=device,
                          in_ndc=False,
                          image_size=torch.tensor([image_size]))




# SoftPhongShader
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)


# silhouette renderer
# TODO rendering error : depth of object is always one
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=silhouette_raster_settings
    ),
    shader=SoftSilhouetteShader(
        #blend_params = blend_params
    )
)

fragments = rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=silhouette_raster_settings
    )


tic3 = time.time()
images = renderer(meshes, R=R_n, T=T_n)

tic4 = time.time()
T = T_n + 0.05* torch.randn_like(T_n); 
#print(T_n, T)
depthes =  fragments(meshes, R = R_n, T=T_n).zbuf
#depthes =  renderer(meshes,  R=R_n, T=T_n)
#print(depthes.shape, depthes[0,82,255,:].cpu().numpy())
depthes_rand =  renderer(meshes, R=R_n, T=T)
tic5 = time.time()

print('create %d cameras takes %.6f secs, create renderer takes %.6f secs, color render takes %.6f, depth render takes %.6f secs'%(len(images),(tic2-tic1),(tic3-tic2),(tic4-tic3), (tic5-tic4)))
plt.figure(figsize=(10, 10))
plt.subplot(2,2,1); plt.imshow(rendered_depth_image);plt.axis("off"); plt.title('Pyrender')
plt.subplot(2,2,2); plt.imshow(depthes[0, ..., 3].cpu().numpy());plt.axis("off");plt.title('Pytorch3D')
plt.subplot(2,2,3); plt.imshow(depthes[0, ..., 3].cpu().numpy() - rendered_depth_image);plt.axis("off");plt.title('Diff')

plt.show()
    
print('Aveg rendering consuming takes %.6f secs'%((time.time()-tic_start)/100))

    



