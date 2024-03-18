import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import cv2

from torch_renderer import *
# colormapping
colormap = plt.cm.viridis
norm = lambda X: (X-X.min())/(X.max()-X.min())


def from_tf_to_matrix(trans, rot):
    trans, rot = np.asarray(trans), np.asarray(rot)
    result = np.eye(4, dtype='float32')
    R = Rotation.from_quat(rot).as_matrix()
    result[:3, :3] = R
    result[:3, 3]  = trans
    return result


# compare to rviz
K_cam11 =  np.array([917.700439453125, 0.0, 646.2421875, 0.0, 0.0, 915.8268432617188, 367.7502136230469, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.], dtype='float32').reshape(4,4)
K_cam12 =  np.array([914.483154296875, 0.0, 645.4754638671875, 0.0, 0., 913.0062866210938, 367.93243408203125, 0.0, 0.0, 0.,0., 1.0, 0.0, 0.0, 1.0, 0.], dtype='float32').reshape(4,4)
ext_cam11 = from_tf_to_matrix([0.008, 0.081, 0.662],[0.813, -0.270, 0.065, 0.511])
ext_cam12 = from_tf_to_matrix([0.233, -0.089, 0.837],[-0.058, 0.869, -0.483, -0.089])

object_trans = np.array([-4.1235436396921584e-05, -0.012795715280304881, 0.08311866360731425], dtype='float32')
object_quat  = np.array([-0.01184704775361205, -0.007628367207312281, 0.15889129486561074, 0.9871955287019907], dtype='float32')

#object_trans += 2 * (np.random.rand(3) - 0.5) * 0.05
#object_quat += 2 * (np.random.rand(4) - 0.5) * 0.05
#object_mat   = from_tf_to_matrix(object_trans, object_quat)


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
object_mat[:3, 3] += 2 * (np.random.rand(3) - 0.5) * 0.05


obj_filename =  "/home/datasets/YCB_DATASETS/models/006_mustard_bottle/textured_simple.obj"

cams = {'11':{'K': K_cam11, 'Ext': extrinsic}, '12':{'K': K_cam12, 'Ext': ext_cam12}}
#cams = {'11':{'K': K_cam11, 'Ext': ext_cam11}, '12':{'K': K_cam12, 'Ext': ext_cam12}}
image_size = (720//4,1280//4)
#image_size = (720,1280)

meshes = load_objs_as_meshes([obj_filename], device=device)



tic_start = time.time()

#for cam in ['11', '12']:
cam = '11'
K = torch.from_numpy(cams[cam]['K']).unsqueeze(0).to(device)
K = K//4

K_3 = K[:,:3,:3].to(device)
K_3[:,2,2] = 1.
#print('K3', K_3)
#camera_pose = np.linalg.inv(object_mat) @ cams[cam]['Ext']
camera_pose = np.dot(cams[cam]['Ext'], object_mat)

object_mat = torch.from_numpy(object_mat).to(device).requires_grad_(True)
cam_ext = torch.from_numpy(cams[cam]['Ext']).to(device)

camera_pose_t = torch.matmul(cam_ext, object_mat)
print("/////////", camera_pose-camera_pose_t.detach().cpu().numpy())

camera_pose = torch.from_numpy(camera_pose).to(device)
#print('tensor camera pose: ', camera_pose[:3,:3], '\n tensor matrix to quat: ', matrix_to_quaternion(camera_pose[:3,:3]) )
#print(quaternion_to_matrix(matrix_to_quaternion(camera_pose[:3,:3])))

R = camera_pose_t[:3, :3][None]
T = camera_pose_t[:3, 3][None]

N = 1 
Rs = R.repeat(N,1,1); ts = T.repeat(N,1); K_3 = K_3.repeat(N,1,1)
meshes = meshes.extend(N).to(device)


depth_renderer = DepthRender(K=K_3, image_size=image_size)
color_renderer = ColorRender(K=K_3, image_size=image_size)

depths_gt = torch.from_numpy(rendered_depth_image)[None]
mask_gt = (depths_gt != 0).bool()

depths_gt = depths_gt.to(device)
mask_gt = mask_gt.to(device)

mseloss = torch.nn.MSELoss()
huberloss = torch.nn.HuberLoss(delta=0.05)
l1loss = torch.nn.L1Loss()

def calc_loss(depth, depth_mask):
    mask = mask_gt 

    sil_loss = l1loss(depth_mask, mask.float())

    depth_gt = torch.masked_select(depths_gt, mask)
    depth = torch.masked_select(depth, mask)

    MSEloss = mseloss(depth, depth_gt)
    hloss = huberloss(depth, depth_gt)

    print("///////////// loss (huber, sil, mse): ", hloss, sil_loss, MSEloss)
    return sil_loss+hloss

optimizer = torch.optim.Adam([object_mat], lr=0.001)

for i in range(200):
    depthes, silheuette = depth_renderer.render(meshes=meshes, R=Rs, tvec=ts, return_silhouette=True)
    #images = color_renderer.render(meshes, R, T)
    loss = calc_loss(depthes, silheuette) 
    loss.backward(retain_graph=True)
    # Save outputs to create a GIF. 
    if i % 10 == 0:
        depth = depthes.detach().squeeze().cpu().numpy()
        diff = (depth-rendered_depth_image)**2
        images = np.hstack((rendered_depth_image, depth, diff))
        cv2.imshow("Diff", cv2.cvtColor(colormap(norm(images)).astype(np.float32), cv2.COLOR_BGR2RGB))
    
        # Check for key press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break


print(depthes, silheuette, images)

plt.figure(figsize=(10, 10))
plt.subplot(2,3,1); plt.imshow(rendered_depth_image);plt.axis("off"); plt.title('Pyrender')
plt.subplot(2,3,2); plt.imshow(depthes[0].detach().cpu().numpy());plt.axis("off");plt.title('Pytorch3D')
plt.subplot(2,3,3); plt.imshow(depthes[0].detach().cpu().numpy() - rendered_depth_image);plt.axis("off");plt.title('Diff')
plt.subplot(2,3,4); plt.imshow(silheuette[0].detach().cpu().numpy());plt.axis("off");plt.title('RGB')
plt.subplot(2,3,5); plt.imshow(images[0].detach().cpu().numpy());plt.axis("off");plt.title('RGB')

plt.show()



