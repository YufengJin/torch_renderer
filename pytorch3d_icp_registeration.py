import os
import sys
import torch
import pytorch3d
import open3d as o3d
import copy
import time
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.transforms import euler_angles_to_matrix
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
Note: 
ICP registeration is deterministic when reference pcd, target pcd, and initial transformation are deterministic. 
the target pcd with less vertices can not achieve better ICP result than the ones with more vertices
"""

def draw_registration_result_o3d(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])


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


def draw_multi_points(pcdList):
    all_pcds = []
    for pcd_tensor in pcdList:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_tensor.cpu().numpy())
        pcd.paint_uniform_color(np.random.rand(3))
        all_pcds.append(pcd)
    o3d.visualization.draw(all_pcds)


def create_register_data(ObjectNums, verts):
    source_pcd = verts.cpu().numpy()
    source_pcd = torch.tensor(source_pcd).type(torch.float32)
    
    # number of vertices
    VertsNums = source_pcd.shape[0]
    source_pcdsList = []
    for i in range(ObjectNums):
        np.random.seed(20)
        #pcdIdx = np.random.choice(VertsNums, size=np.random.randint(low=1500, high=2000))
        pcdIdx = np.random.choice(VertsNums, size=2000,)
        pcd = source_pcd[pcdIdx, :]
    
        source_pcdsList.append(pcd)
    
    # create pytorch3d pointcloud
    source_pcds = Pointclouds(source_pcdsList)
    
    
    target_pcdsList = copy.deepcopy(source_pcdsList)
    #target_pcdsList = tmp.points_list()
    
    # create gt transformations
    transTensors = torch.zeros(ObjectNums, 3)
    rotTensors = torch.zeros(ObjectNums, 3, 3)
    
    for i in range(ObjectNums):
        np.random.seed(5)
        #trans = torch.tensor([0.005, 0.007, 0.005])
        trans = torch.randn(3)*0.5
        trans[2] = 0.
        transTensors[i] = trans
        #rot = torch.tensor([0,0,np.pi/4])
        rot = torch.rand(3)*np.pi/4
        rotTensors[i]   = euler_angles_to_matrix(rot, convention="XYZ")
    
    #transforms[:,:3, :3] = euler_angles_to_matrix(torch.tensor([0,0,np.pi/4]), convention="XYZ")
    #transforms[:,:3, 3] = torch.tensor([0.00 , 0.0, .01])
    
    # transform target pcd
    target_pcds = [] 
    for i, pcd_tensor in enumerate(target_pcdsList):
        target_pcd = pcd_tensor.cpu().numpy()
        target_pcd = np.dot(rotTensors[i].cpu().numpy(), target_pcd.T)
        target_pcd = target_pcd.T + transTensors[i].cpu().numpy()
        
        # random select target pcd
        if True:
            #np.random.seed(10)
            pcdIdx = np.random.choice(target_pcd.shape[0], size=np.random.randint(low=200, high=800))
            #pcdIdx = np.random.choice(target_pcd.shape[0], size=1000, )#np.random.randint(low=200, high=500))
            target_pcd = target_pcd[pcdIdx, :]
        
        # add noise for target pcd
        if True:
            target_pcd += np.random.randn(target_pcd.shape[0], 3)*0.0001

        if True:
            center_pcd = target_pcd.mean(axis=0)
            mask_x = target_pcd[:,0] > center_pcd[0]
            target_pcd = target_pcd[mask_x]

    
        target_pcds.append(torch.tensor(target_pcd).type(torch.float32))
    
    target_pcds = Pointclouds(target_pcds)
    
    #draw_multi_points(target_pcds.points_list())
    
    
    
    Rs = torch.eye(3).repeat(ObjectNums,1).reshape(ObjectNums, 3, 3)
    ts = transTensors + torch.randn(transTensors.shape)*0.05
    #ts = torch.zeros(3).repeat(ObjectNums).reshape(ObjectNums, 3)
    ss = torch.ones(1).repeat(ObjectNums)
    return source_pcds, target_pcds, Rs, ts, ss, transTensors, rotTensors

def ICP_on_GPU(source_pcds, target_pcds, Rs, ts, ss):
    # convert to cuda
    sources = source_pcds.to(device)
    
    tic_init = time.time()
    targets = target_pcds.to(device)
    init_trans = (Rs.to(device), ts.to(device), ss.to(device))
    
    tic= time.time()

    # compute chamfer loss 
    loss, loss_normals = pytorch3d.loss.chamfer_distance(sources.points_padded(), targets.points_padded(), batch_reduction=None)
    print("loss takes %.6f: "%(time.time()-tic), loss)
    
    #converged, rmse, transed_src, RTs, t_history = pytorch3d.ops.iterative_closest_point(sources, targets, init_trans, max_iterations=100)
    converged, rmse, transed_src, RTs, t_history = pytorch3d.ops.iterative_closest_point(sources.points_padded(), targets.points_padded(), init_trans, max_iterations=100)
    dt = time.time()-tic
    print("[PYTORCH] ICP for %d objects, takes %.6f s running on GPU, data flows from cpu to GPU takes %.6f "%(len(sources.points_list()), dt, (tic-tic_init)))
    #print("[PYTORCH] registeration results, rmse arranged from worst to best: \n", np.sort(rmse.cpu().numpy())[::-1])
    print("[PYTORCH] registeration results, rmse: \n", rmse.cpu().numpy())
    #print("rmse: ", rmse)
    #print("converged: ", converged)
    #print("Rt_hist: ", t_history[-1].R, t_history[-1].T)
    #print("Rt_gt: ", rotTensors,transTensors )
    #all_pcds = []
    #for i, pcd in enumerate(transed_src.points_list()):
    #for i, pcd in enumerate(transed_src):
    #    print("transformed source pcd %d, with %d vertices. "%(i+1, pcd.shape[0]))
    #    all_pcds.append(pcd)
    #for pcd in target_pcds.points_list():
    #    all_pcds.append(pcd)
    return converged, rmse, transed_src, RTs, t_history, dt

def quaternion_distance(q1, q2):
    temp = (q1*q2).sum()**2
    return 1-temp, np.arccos(2*temp-1)

def ICP_on_CPU(source_pcds, target_pcds, ts, threshold=0.01 ,debug=False):
    # open3d icp registeration
    import open3d as o3d
    
    # icp for all object
    open3d_data = {
    "source": [],
    "target": [],
    "transform": [],
    "source_trans" : []
    }
    
    # convert pcd tensor to open3d correspionding PointCloud
    for i, (source_t, target_t) in enumerate(zip(source_pcds.points_list(), target_pcds.points_list())):
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_t.cpu().numpy())
        target.points = o3d.utility.Vector3dVector(target_t.cpu().numpy())
    
        transform =  np.eye(4)
        transform[:3, 3] = ts[i].cpu().numpy()
    
        open3d_data["source"].append(source)
        open3d_data["target"].append(target)
        open3d_data["transform"].append(transform)
    
    tic = time.time()
    
    results = []
    for (source, target, init_trans) in list(zip(open3d_data["source"], open3d_data["target"], open3d_data["transform"])):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        results.append(reg_p2p)
        if debug: 
            print("Apply point-to-point ICP, result:\n", reg_p2p)
            draw_registration_result_o3d(source, target, reg_p2p.transformation)
        source_trans = source.transform(reg_p2p.transformation)
        pcd = np.asarray(source_trans.points)
        open3d_data["source_trans"].append(torch.tensor(pcd).type(torch.float32))

    dt = time.time() - tic
    inlier_rmses = np.asarray([reg.inlier_rmse for reg in results])
    transforms   = np.asarray([reg.transformation for reg in results])
    print("[Open3D] ICP for %d objects, takes %.6f s running on CPU"%(len(open3d_data['source']) ,dt))
    print("[Open3D] registeration results, rmse: \n", inlier_rmses)
    #print("[Open3D] registeration results, inlier rmse arranged from worst to best: \n", np.sort(inlier_rmses)[::-1])
    return open3d_data["source_trans"], transforms , dt

def time_running_statistic(ObjectNumsList, verts):
    time_takes = [] # save in tuple (numbers of object, times runs in GPU, time runs in CPU)
    for ObjectNums in ObjectNumsList:
        # create all src targ pcd and init transform for registeration
        source_pcds, target_pcds, Rs, ts, ss, transTensors, rotTensors = create_register_data(ObjectNums, verts)
    
        converged, rmse, transed_src, RTs, t_history, dt_GPU = ICP_on_GPU(source_pcds, target_pcds, Rs, ts, ss)
        transed_src_o3d, RTs_o3d, dt_CPU = ICP_on_CPU(source_pcds, target_pcds, ts)
    
        time_takes.append((ObjectNums, dt_GPU, dt_CPU))
    
    
    _, dts_GPU, dts_CPU = list(zip(*time_takes))
    plt.subplot(); plt.plot(ObjectNumsList, dts_GPU, label="GPU"); plt.plot(ObjectNumsList, dts_CPU, label="CPU"); 
    plt.xlabel("Number of Objects"); plt.ylabel("time in seconds");
    plt.title("Comparison of ICP running on CPU and GPU");
    plt.legend()
    plt.show()

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# Load the dolphin mesh.
#trg_obj = os.path.join('./data/cow_mesh', 'cow.obj')
trg_object = '006_mustard_bottle'
trg_obj = os.path.join('../YCB_Video_Dataset/models/',trg_object ,'textured.obj')


# We read the target 3D model using load_obj
verts, faces, aux = load_obj(trg_obj)
#time_running_statistic(np.arange(1,100,2), verts)

ObjectNums = 300 

# create all src targ pcd and init transform for registeration
source_pcds, target_pcds, Rs, ts, ss, transTensors, rotTensors = create_register_data(ObjectNums, verts)

if True:
# visualization all pcds
   source = []
   for i, src_pcd in enumerate(source_pcds.points_list()):
       source.append(src_pcd + ts[i])
   draw_registration_result_tensor(source, target_pcds.points_list()) 


converged, rmse, transed_src, RTs, t_history, dt_GPU = ICP_on_GPU(source_pcds, target_pcds, Rs, ts, ss)
transed_src_o3d, RTs_o3d, dt_CPU = ICP_on_CPU(source_pcds, target_pcds, ts)

print("[OPEN3D] ICP for %d objects, takes %.6f s running on CPU"%(ObjectNums, dt_CPU))
# get vertices number for target point
target_pcd_vertices = np.asarray([pcd.cpu().numpy().shape[0] for pcd in target_pcds.points_list()])

trans_gt = transTensors.cpu().numpy()
rot_gt = rotTensors.cpu().numpy()

trans_hist = np.asarray([t_history[i].T.cpu().numpy() for i in range(len(t_history))])
rot_hist = np.asarray([t_history[i].R.cpu().numpy() for i in range(len(t_history))])

trans_diff = trans_hist - trans_gt


from scipy.spatial.transform import Rotation as R
losses = {'trans': np.zeros((len(t_history), ObjectNums)),
          'theta'  : np.zeros((len(t_history), ObjectNums))}
losses['vertices'] = target_pcd_vertices
print("target pcd vertices: ", losses['vertices'])        

# transforms error from pytorch
for i in range(len(t_history)):
    for idx in range(ObjectNums):
        quat_gt = R.from_matrix(rot_gt[idx]).as_quat()
        quat_y  = R.from_matrix(rot_hist[i][idx].T).as_quat()
        L2 = ((trans_hist[i][idx] - trans_gt[idx])**2).sum()**0.5
        _, theta = quaternion_distance(quat_gt, quat_y)
        losses['trans'][i][idx] = L2
        losses['theta'][i][idx] = theta

o3d_final_losses = {"trans": np.zeros(ObjectNums),
                    "theta": np.zeros(ObjectNums)}
# transforms error from o3d
for idx in range(ObjectNums):
    quat_gt = R.from_matrix(rot_gt[idx]).as_quat()
    quat_y  = R.from_matrix(RTs_o3d[idx][:3,:3]).as_quat()
    L2 = ((RTs_o3d[idx][:3,3] - trans_gt[idx])**2).sum()**0.5
    _, theta = quaternion_distance(quat_gt, quat_y)
    o3d_final_losses['trans'][idx] = L2
    o3d_final_losses['theta'][idx] = theta

plt.subplot(2,1,1); plt.plot(losses['trans'], label=losses['vertices']);plt.title("cartesian distance error"); plt.xlabel("iters"); plt.ylabel('m'); plt.legend()
plt.subplot(2,1,2); plt.plot(losses['theta'], label=losses['vertices']);plt.title("orientation error"); plt.xlabel("iters"); plt.ylabel('rad'); plt.legend()
plt.show()

print("-------------------------------- GPU -------------------------------------------------")
print("Remaining translation error L2 : \n{}, \norientation error theta (in rads):\n {} \norientation error theta (in degrees):\n{}".format(losses['trans'][-1], losses['theta'][-1], losses['theta'][-1]/np.pi*180))
print("\n\n\n------------------------------------CPU---------------------------------------------")
print("Remaining translation error L2 : \n{}, \norientation error theta (in rads):\n {} \norientation error theta (in degrees):\n{}".format(o3d_final_losses['trans'], o3d_final_losses['theta'], o3d_final_losses['theta']/np.pi*180))
#print(trans_gt.shape, rot_gt.shape, trans_hist.shape, rot_hist.shape)


plt.subplot(1,2,1);
plt.scatter(x=list(range(1,ObjectNums+1)), y=o3d_final_losses['trans'],c='b',s=5,label="open3d")
plt.scatter(x=list(range(1,ObjectNums+1)), y=losses['trans'][-1], c='r',s=5, label="pytorch")
plt.xlabel("Object index"); plt.ylabel("m");
plt.title("Difference in translation error (Open3d & Pytorch)");
plt.legend()

plt.subplot(1,2,2);
plt.scatter(x=list(range(1,ObjectNums+1)), y=o3d_final_losses['theta'],c='b',s=5, label="open3d")
plt.scatter(x=list(range(1,ObjectNums+1)), y=losses['theta'][-1], c='r',s=5, label="pytorch")
plt.xlabel("Object index"); plt.ylabel("rads");
plt.title("Difference in orientation error (Open3d & Pytorch)");
plt.legend()
plt.show()

draw_registration_result_tensor(transed_src, target_pcds.points_list())
draw_registration_result_tensor(transed_src_o3d, target_pcds.points_list())



