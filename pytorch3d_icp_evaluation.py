"""
   batch icp evaluation
"""
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
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from matplotlib.patches import Ellipse

TEST_OBJECT_ID = 36

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


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, axis=None, label=True, title=None, ax=None, xlim=None, ylim=None):
    assert axis is not None, "axises are required"
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    if axis is not None:
        ax.set_xlabel(axis[0])
        ax.set_ylabel(axis[1])
    
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, ax=ax)
    if title: 
        ax.set_title(title)

    
class ICPTensorEvalutor:
    def __init__(self):
    # Set the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            print("WARNING: CPU only, this will be slow!")
        
        ycb_video_path = "../YCB_Video_Dataset/models/"

        self.ref_models = {}
        self.load_ycb_models(ycb_video_path)
        
        self.icp_eva_test()
    
    
    def load_ycb_models(self, ycb_video_path, ref_obj_vertices = 500):
        objIdList = os.listdir(ycb_video_path)

        for objIdx in objIdList:
            objId = int(objIdx.split("_")[0])
            if objId != TEST_OBJECT_ID: continue
            # TODO  load vertice, convert into tensors, TensorList : [Tensor(vertices, 3) for one object, ... ]
            objPath = os.path.join(ycb_video_path, objIdx, 'textured.obj')
            # We read the target 3D model using load_obj
            verts, faces, aux = load_obj(objPath)
            objMesh = Meshes(verts=[verts], faces=[faces.verts_idx], )
            downsampled_pcd = sample_points_from_meshes(objMesh, num_samples=ref_obj_vertices)

            # pcd saved in tensor, with Size([1, verticesNum, 3])
            self.ref_models[objId] = downsampled_pcd.to(self.device)
            print('PointClouds of %d object with %d vertices loaded in Memory'%(objId, ref_obj_vertices))
            
    def xyz_rpy_numpy_to_tensor(self, xyz_ts, rpy_ts, sampleNum):
        xyz_ts = torch.tensor(xyz_ts).to(self.device)
        xyz_ts = xyz_ts.reshape(sampleNum, 1, 3)
        rpy_ts   = torch.tensor(rpy_ts).to(self.device)
        rot_ts   = euler_angles_to_matrix(rpy_ts, convention="XYZ")
        return xyz_ts, rot_ts
    
    def run_icp_evaluation(self, objId, target_pcd, gmmModel = GaussianMixture ,sampleNum=400, trans_scale =0.05, iterations=10, gmm_component_n=4, log=True):
        # TODO uniform Sample translation and orientation(x,y,z,r,p,y); or GMM model with gaussian disturbition model
        assert isinstance(target_pcd, torch.Tensor), 'target pointcloud should be converted to a Tensor'

        gmm_hist = list()
        if target_pcd.device != self.device: 
           print("target pointCloud was on CPU, converted to GPU")
           target_pcd = target_pcd.to(self.device)

        if len(target_pcd.shape) == 2: 
             target_pcd = target_pcd.unsqueeze(0)

        pcd_mean = target_pcd.squeeze().cpu().numpy().mean(axis=0) # x,y,z
        
        # translation sample from gaussian distibution with mean = means of target pointcloud and +/- 0.15 as tolerance
        init_transes = np.random.normal(loc=pcd_mean, scale=np.ones(3)*trans_scale, size=(sampleNum,3)).astype(np.float32)
        # uniform orientation sample between -/+ np.pi 
        init_rpy    = np.random.uniform(-np.pi, np.pi, size=sampleNum*3).astype(np.float32).reshape(-1,3)

        if log: 
            X_hist = {'translation': [],
                      'orientation': []} 
            X_hist['translation'].append(init_transes)
            X_hist['orientation'].append(init_rpy)
        
        # TODO diag or full, whether translation and orientation is independent with each other?, spherical, diag, full, tied, in increase order of performace
        # k-means++ best init params
        gmm = gmmModel(gmm_component_n, covariance_type='diag', random_state=0, init_params='k-means++')
         
        # convert translation and orientation to tensor
        trans_ts, rot_ts = self.xyz_rpy_numpy_to_tensor(init_transes, init_rpy, sampleNum)
        # torch.TensorFloat

        ref_pcds  = self.ref_models[objId].repeat(sampleNum, 1, 1)
 
        ref_pcds = transform_pcd_tensors(ref_pcds, trans_ts, rot_ts)
        tar_pcds = target_pcd.repeat(sampleNum, 1, 1)

        # chamfer loss 
        tic_st = time.time()

        loss = kaolin.metrics.pointcloud.chamfer_distance(ref_pcds, tar_pcds)
        #loss, loss_normals = pytorch3d.loss.chamfer_distance(ref_pcds, tar_pcds, batch_reduction=None)
        print("loss takes %.6f: "%(time.time()-tic_st))
        
        xyz_rpy_np = np.hstack((init_transes, init_rpy))
        # get good samples with low chamfer loss, if nans always at the end of list
        goodIdx = np.argsort(loss.cpu().numpy())[:100]
        print(f"max of loss: {loss.cpu().numpy().max()}")
        X = xyz_rpy_np[goodIdx]
        gmm.fit(X)
        print(f"gmm means: \n{gmm.means_}\n cov: \n{gmm.covariances_}")
        gmm_hist.append(copy.deepcopy(gmm))
        # EM algorithm
        for i in range(iterations):
            xyz_rpy_np, labels = gmm.sample(sampleNum)
       
            xyz_rpy_np = xyz_rpy_np.astype(np.float32)
            xyz_np = xyz_rpy_np[:, :3]
            rpy_np = xyz_rpy_np[:, 3:]

            if log: 
                X_hist['translation'].append(xyz_np)
                X_hist['orientation'].append(rpy_np)

            trans_ts, rot_ts = self.xyz_rpy_numpy_to_tensor(xyz_np, rpy_np, sampleNum)
            ref_pcds  = self.ref_models[objId].repeat(sampleNum, 1, 1)

            ref_pcds = transform_pcd_tensors(ref_pcds, trans_ts, rot_ts)
            tar_pcds = target_pcd.repeat(sampleNum, 1, 1)

            # chamfer loss 
            tic = time.time()
            #loss, loss_normals = pytorch3d.loss.chamfer_distance(ref_pcds, tar_pcds, batch_reduction=None)
            loss = kaolin.metrics.pointcloud.chamfer_distance(ref_pcds, tar_pcds)
            print("loss takes %.6f: "%(time.time()-tic))
            goodIdx = np.argsort(loss.cpu().numpy())[:50]
            print(f"iter {i}:    max loss: {loss.cpu().numpy().max()}")
            X = xyz_rpy_np[goodIdx]
            gmm.fit(X)
            print(f"gmm means: \n{gmm.means_}\ncov: \n{gmm.covariances_}")
            gmm_hist.append(copy.deepcopy(gmm))

        print(f"{iterations} iterations takes {time.time()-tic_st:.6f} s")
        if log:
            xlimt = list()  
            ylimt = list()
            for i in range(iterations+1):
                xyz_all = np.asarray(X_hist['translation'][i]).reshape(-1,3)
                rpy_all = np.asarray(X_hist['orientation'][i]).reshape(-1,3)
                XY = xyz_all[:,:2]
                YZ = xyz_all[:,1:]
                XZ = xyz_all[:,[0,2]]
                
                RP = rpy_all[:,:2]
                PY = rpy_all[:,1:]
                RY = rpy_all[:,[0,2]]

                if len(xlimt) == len(ylimt) == 0:
                    for xy_d in [XY, YZ, XZ]:
                        assert xy_d.shape[1] == 2, "scatter data must be 2D" 
                        xlimt.append((xy_d[:,0].min(), xy_d[:,0].max()))
                        ylimt.append((xy_d[:,1].min(), xy_d[:,1].max()))

                (gmm_xy, gmm_yz, gmm_xz, gmm_rp, gmm_py, gmm_ry) = [gmmModel(gmm_component_n, covariance_type='diag', random_state=0) for i in range(6)]
                gmm_xy.fit(XY)
                gmm_yz.fit(YZ)
                gmm_xz.fit(XZ)
                gmm_rp.fit(RP)
                gmm_py.fit(PY)
                gmm_ry.fit(RY)

                fig, ax =  plt.subplots(2,3, figsize=(15, 10))
                plot_gmm(gmm_xy, XY, ax=ax[0][0], axis=['x','y'],title='GMM of xy(xyz) translation',xlim=xlimt[0], ylim=ylimt[0])
                plot_gmm(gmm_yz, YZ, ax=ax[0][1], axis=['y','z'],title='GMM of yz(xyz) translation',xlim=xlimt[1], ylim=ylimt[1])
                plot_gmm(gmm_xy, XZ, ax=ax[0][2], axis=['x','z'],title='GMM of xz(xyz) translation',xlim=xlimt[2], ylim=ylimt[2])
                plot_gmm(gmm_rp, RP, ax=ax[1][0], axis=['r','p'],title='GMM of rp(rpy) orientation',xlim=(-6,6), ylim=(-6,6))
                plot_gmm(gmm_py, PY, ax=ax[1][1], axis=['p','y'],title='GMM of py(rpy) orientation',xlim=(-6,6), ylim=(-6,6))
                plot_gmm(gmm_ry, RY, ax=ax[1][2], axis=['r','y'],title='GMM of ry(rpy) orientation',xlim=(-6,6), ylim=(-6,6))
                plt.suptitle(f"OBJECT ID: {TEST_OBJECT_ID:03d}")
                #plt.show()
                plt.savefig(f'PUResults/{TEST_OBJECT_ID:03d}_{time.strftime("%Y-%d-%m-%H-%M-%S")}_iter_{i}.png', dpi=400, format='png')
        return gmm, gmm_hist
    
        
    def icp_eva_test(self):
        target_id = TEST_OBJECT_ID 
        target_pcd = copy.deepcopy(self.ref_models[target_id])
        
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
                center_pcd = pcd_np.mean(axis=0)
                mask_x = pcd_np[:,0] > center_pcd[0]
                pcd_np = pcd_np[mask_x]

            pcd = torch.tensor(pcd_np.astype(np.float32)).unsqueeze(0)


        if True:
            draw_points_list(pcd)

        print(f"translation error in meters: {trans.cpu().numpy()}, orientation error in rads: {rot.cpu().numpy()}")
        tic = time.time()
        gmm, gmm_hist = self.run_icp_evaluation(target_id, pcd)

        xyz_rpy = gmm.means_.astype(np.float32)
        print("Time consuming : ", time.time()-tic)

        for i in range(2):
            trans = xyz_rpy[i][:3]
            rot   = xyz_rpy[i][3:]

            trans = torch.tensor(trans)
            rot   = torch.tensor(rot)
            R = euler_angles_to_matrix(rot, convention="XYZ")
            trans = trans.reshape(1,3)[None, ...]
            R = R[None, ...]

            transed_pcd = transform_pcd_tensors(target_pcd, trans, R)
            
            draw_registration_result_tensor(pcd, transed_pcd)


if __name__ == '__main__':

    icpEva = ICPTensorEvalutor()



