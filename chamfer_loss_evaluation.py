#
# code for evaluate the quality of chamfer loss running on GPU
#
#
import os
import sys
import torch
import pytorch3d
import open3d as o3d
import copy
import time
import kaolin
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.loss import chamfer_distance
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation as R


def quaternion_distance(q1, q2):
    temp = (q1*q2).sum()**2
    return 1-temp, np.arccos(2*temp-1)

def draw_registration_result_tensor(source, target):
    source_temp = o3d.geometry.PointCloud()
    target_temp = o3d.geometry.PointCloud()

    source_points = []
    target_points = []
    for pcd_tensor in source:
        points = pcd_tensor.cpu().squeeze().numpy()
        #points = points[np.all(points!=np.zeros(3))]
        source_points.append(points)

    source_points = np.vstack(source_points)
    source_temp.points = o3d.utility.Vector3dVector(source_points)

    for pcd_tensor in target:
        points = pcd_tensor.cpu().squeeze().numpy()
        #points = points[np.all(points!=np.zeros(3))]
        target_points.append(points)

    target_points = np.vstack(target_points)
    target_temp.points = o3d.utility.Vector3dVector(target_points)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw([source_temp, target_temp])

def draw_points_list(pcdList):
    all_pcds = []
    for pcd_tensor in pcdList:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_tensor.cpu().numpy())
        pcd.paint_uniform_color(np.random.rand(3))
        all_pcds.append(pcd)
    o3d.visualization.draw(all_pcds)

def transform_pcd_tensors(pcds, ts, Rs):
    assert (len(pcds.shape) == len(ts.shape) == len(Rs.shape) == 3) \
           and (pcds.shape[0] == ts.shape[0] == Rs.shape[0]), \
           "PointClouds should with Size([N,vertices,3]), trans with Size([N,1,3]) and rots with Size([N,3,3]) "
    verts_num = pcds.shape[1]
    pcds = torch.matmul(Rs, torch.transpose(pcds, 1,2))
    pcds = torch.transpose(pcds, 2,1) + ts.repeat(1, verts_num, 1)
    return pcds

class ChamferLossTest:
    def __init__(self, object_id):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            print("WARNING: CPU only, this will be slow!")

        ycb_video_path = "../YCB_Video_Dataset/models/"

        objPath = os.path.join(ycb_video_path, object_id, 'textured.obj')
        # We read the target 3D model using load_obj
        verts, faces, aux = load_obj(objPath)
        objMesh = Meshes(verts=[verts], faces=[faces.verts_idx], )
        downsampled_pcd = sample_points_from_meshes(objMesh, num_samples=1000)

        # pcd saved in tensor, with Size([1, verticesNum, 3])
        self.ref_model = downsampled_pcd.to(self.device)

        self.main()

    def xyz_rpy_numpy_to_tensor(self, xyz_ts, rpy_ts, sampleNum):
        xyz_ts = torch.tensor(xyz_ts).to(self.device)
        xyz_ts = xyz_ts.reshape(sampleNum, 1, 3)
        rpy_ts   = torch.tensor(rpy_ts).to(self.device)
        rot_ts   = euler_angles_to_matrix(rpy_ts, convention="XYZ")
        return xyz_ts, rot_ts

    def chamfer_loss_evaluation(self, target_pcd, t_gt, R_gt, sampleNum=1000, trans_scale=0.05):
        if target_pcd.device != self.device:
           print("target pointCloud was on CPU, converted to GPU")
           target_pcd = target_pcd.to(self.device)

        if len(target_pcd.shape) == 2:
             target_pcd = target_pcd.unsqueeze(0)

        print(f"translation gt: {t_gt} and orientation gt: {R_gt}") 
        pcd_mean = t_gt.cpu().numpy().ravel() # x,y,z
        # translation sample from gaussian distibution with mean = means of target pointcloud and +/- 0.15 as tolerance
        init_transes = np.random.normal(loc=pcd_mean, scale=np.ones(3)*trans_scale, size=(sampleNum,3)).astype(np.float32)
        # uniform orientation sample between -/+ np.pi 
        init_rpy    = np.random.uniform(-np.pi, np.pi, size=sampleNum*3).astype(np.float32).reshape(-1,3)
        trans_ts, rot_ts = self.xyz_rpy_numpy_to_tensor(init_transes, init_rpy, sampleNum)
        ref_pcds  = self.ref_model.repeat(sampleNum, 1, 1)

        ref_pcds = transform_pcd_tensors(ref_pcds, trans_ts, rot_ts)
        tar_pcds = target_pcd.repeat(sampleNum, 1, 1)

        #loss = kaolin.metrics.pointcloud.chamfer_distance(ref_pcds, tar_pcds)
        loss, loss_normals = pytorch3d.loss.chamfer_distance(ref_pcds, tar_pcds, batch_reduction=None)
        print(f"loss : {loss}")

        chamfer_losses = loss.cpu().numpy()
        # compute cartesian distance and quaternion distance

        trans_nps, rot_nps = trans_ts.cpu().numpy(), rot_ts.cpu().numpy()
        
        assert trans_nps.shape[0] == rot_nps.shape[0] == len(chamfer_losses)

        losses = {'trans': list(),
                  'theta': list()
                  }
        trans_gt, rot_gt = t_gt.cpu().numpy().reshape(1,3), R_gt.cpu().numpy().reshape(3,3)
        for idx in range(trans_nps.shape[0]):
            quat_gt = R.from_matrix(rot_gt).as_quat()
            quat_y  = R.from_matrix(rot_nps[idx]).as_quat()
            L2 = ((trans_nps[idx] - trans_gt)**2).sum()**0.5
            _, theta = quaternion_distance(quat_gt, quat_y)
            losses['trans'].append(L2)
            losses['theta'].append(theta)


        plt.figure(figsize=(10,10))
          
        from matplotlib.colors import hsv_to_rgb
        h = 1. - chamfer_losses / chamfer_losses.max() # normalize to 0~1 1 with lower chamfer loss
        colors = h.repeat(3).reshape(-1,3)
        colors = hsv_to_rgb(colors)
        plt.scatter(losses['trans'], losses['theta'], c= colors)
        plt.colorbar()
        plt.show()
        
        


    def main(self):
        target_pcd = copy.deepcopy(self.ref_model)

        target_pcd = target_pcd.to(torch.device("cpu"))
        # add transformation on target_pcd 
        trans = torch.randn(3)*0.5
        #rot = torch.tensor([0,0,np.pi/4])
        rot = torch.rand(3)*np.pi/4
        R = euler_angles_to_matrix(rot, convention="XYZ")

        trans = trans.reshape(1,3)[None, ...]
        R = R[None, ...]
        # pcd with Size([1, vertices, 3]), trans with Size([1, 1, 3]), 

        pcd = transform_pcd_tensors(target_pcd, trans, R)

        if True:
            # add gaussian noise and downsample points
            pcd_np = pcd.squeeze().cpu().numpy()

            # random select target pcd
            pcdIdx = np.random.choice(pcd_np.shape[0], size=1000) #np.random.randint(low=200, high=800))
            pcd_np = pcd_np[pcdIdx, :]

            # add noise for target pcd
            pcd_np += np.random.randn(pcd_np.shape[0], 3)*0.0001

        if True:
            # crop pcd 
            center_pcd = pcd_np.mean(axis=0)
            mask_x = pcd_np[:,0] > center_pcd[0]
            pcd_np = pcd_np[mask_x]

            pcd = torch.tensor(pcd_np.astype(np.float32)).unsqueeze(0)

        if True:
            draw_points_list(pcd)

        
        self.chamfer_loss_evaluation(pcd, trans, R)


if __name__ == '__main__':
    clt = ChamferLossTest("006_mustard_bottle")
