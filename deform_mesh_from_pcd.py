import os
import sys
import torch
import pytorch3d
import os
import torch
import open3d as o3d
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

def query_vertex_color_from_o3d_triMesh(verts, mesh):
    ori_verts = np.asarray(mesh.vertices)
    idxes = []
    for vert in verts:
        idx = np.where(np.all(ori_verts==vert, axis=1))[0]
        idxes.append(idx[0])

    verts_color = np.asarray(mesh.vertex_colors)

    return verts_color[idxes, :]

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# Load the dolphin mesh.
#trg_obj = os.path.join('./data/cow_mesh', 'cow.obj')
trg_object = '021_bleach_cleanser'
trg_obj = os.path.join('../YCB_Video_Dataset/models/',trg_object ,'textured.obj')
#trg_obj = "/hri/localdisk/yjin/intro_ros_ws/src/scene_understanding/scripts/scene_understanding/block.obj"

# We read the target 3D model using load_obj
verts, faces, aux = load_obj(trg_obj)

# verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
# faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
# For this tutorial, normals and textures are ignored.
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
# (scale, center) will be used to bring the predicted mesh to its original center and scale
# Note that normalizing the target mesh, speeds up the optimization but is not necessary!
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale

# We construct a Meshes structure for the target mesh
trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

# We initialize the source shape to be a sphere of radius 1
#src_mesh = ico_sphere(4, device)
src_object = '006_mustard_bottle'
src_obj = os.path.join('../YCB_Video_Dataset/models/', src_object ,'textured_simple_colored.obj')

# We read the target 3D model using load_obj
src_verts, src_faces, src_aux = load_obj(src_obj)

# get verts_color from src mesh
src_triangle_mesh = o3d.io.read_triangle_mesh(src_obj)

verts_colors = query_vertex_color_from_o3d_triMesh(src_verts.numpy(), src_triangle_mesh) 

# verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
# faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
# For this tutorial, normals and textures are ignored.
src_faces_idx = src_faces.verts_idx.to(device)
src_verts = src_verts.to(device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
# (scale, center) will be used to bring the predicted mesh to its original center and scale
# Note that normalizing the target mesh, speeds up the optimization but is not necessary!
src_center = src_verts.mean(0)
src_verts = src_verts - src_center
src_scale = max(src_verts.abs().max(0)[0])
src_verts = src_verts / src_scale

# We construct a Meshes structure for the target mesh
src_mesh = Meshes(verts=[src_verts], faces=[src_faces_idx])



def plot_pointcloud(mesh, iteration):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter3D(x, -y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.view_init(15, 0)
    plt.title('iter: %d' % iteration)
    plt.savefig('morphy_inter_results/img%03d_15_00.png'%(iteration//50))
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter3D(x, -y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.view_init(15, 90)
    plt.title('iter: %d' % iteration)
    plt.savefig('morphy_inter_results/img%03d_15_90.png'%(iteration//50))
    plt.close()


# %matplotlib notebook
#plot_pointcloud(trg_mesh, "Target mesh")
#plot_pointcloud(src_mesh, "Source mesh")

# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters is equal to the total number of vertices in src_mesh
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# The optimizer
optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)



# Number of optimization steps
Niter = 4000
# Weight for the chamfer loss
w_chamfer = 1.0 
# Weight for mesh edge loss
w_edge = 1.0 
# Weight for mesh normal consistency
w_normal = 0.01 
# Weight for mesh laplacian smoothing
w_laplacian = 0.1 
# Plot period for the losses
plot_period = 200
loop = tqdm(range(Niter))

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []


for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()
    
    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    
    # We sample 5k points from the surface of each mesh 
    sample_trg = sample_points_from_meshes(trg_mesh, 1000)
    sample_src = sample_points_from_meshes(new_src_mesh, 1000)
    
    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)
    
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    
    # Weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
    
    # Print the losses
    loop.set_description('total_loss = %.6f' % loss)
    
    # Save the losses for plotting
    chamfer_losses.append(float(loss_chamfer.detach().cpu()))
    edge_losses.append(float(loss_edge.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))
    
    # Plot mesh
    if i % plot_period == 0:
        plot_pointcloud(new_src_mesh, i)
        
        # save intermediate obj
        with torch.no_grad():
            # Fetch the verts and faces of the final predicted mesh
            verts, faces = new_src_mesh.get_mesh_verts_faces(0)
            
            # Scale normalize back to the original target size
            rescaled_verts = verts * scale + center
            
            # Store the predicted mesh using save_obj
            obj_path = os.path.join('./morphy_results', 'iter_%05d.obj'%i)
            save_obj(obj_path, rescaled_verts, faces)
    # Optimization step
    loss.backward()
    optimizer.step()




fig = plt.figure(figsize=(13, 5))
ax = fig.gca()
ax.plot(chamfer_losses, label="chamfer loss")
ax.plot(edge_losses, label="edge loss")
ax.plot(normal_losses, label="normal loss")
ax.plot(laplacian_losses, label="laplacian loss")
ax.legend(fontsize="16")
ax.set_xlabel("Iteration", fontsize="16")
ax.set_ylabel("Loss", fontsize="16")
ax.set_title("Loss vs iterations", fontsize="16");



# Fetch the verts and faces of the final predicted mesh
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# Scale normalize back to the original target size
final_verts = final_verts * scale + center

# Store the predicted mesh using save_obj
final_obj = os.path.join('./', 'pcd_morphy_final.obj')
#save_obj(final_obj, final_verts, final_faces)


vertices = final_verts.detach().cpu().numpy()
faces    = final_faces.cpu().numpy()

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.vertex_colors = o3d.utility.Vector3dVector(verts_colors)

o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh(final_obj, mesh)
