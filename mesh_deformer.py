import os
import sys
import torch
import pytorch3d
# load and save mesh
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
# loss funtion
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    AmbientLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
)
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# for image grid visualization
from utils.plot_image_grid import image_grid

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
    plt.show()


class TheCreator:
    def __init__(self, target_object='../YCB_Video_Dataset/models/021_bleach_cleanser/textured.obj', 
                       #source_object='../YCB_Video_Dataset/models/006_mustard_bottle/textured_simple_colored.obj'):
                       source_object='./presentation/after/pcd_morphy_final.obj'):
        # Setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        # Load the target and source meshes with only one object in list. To extend meshes use Meshes.extend(numbers)
        self.trg_mesh, self.offset, self.scale = self.load_and_scale_mesh(target_object)
        self.src_mesh, _, _ = self.load_and_scale_mesh(source_object)
        
        # TODO add rgb parameters to train after deform
        # deform verts, and rgb to train
        self.deform_verts = torch.full(self.src_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        
        # train
        self.geometry_train()

        self.deform_verts.requires_grad_(False)
        
        # create rgb images for color train
        self.color_train(target_object)

    def color_train(self, target_path, num_views=10, Niter = 10000, plot_period=2500, num_views_per_iteration = 5):
        #load textured meshes
        print("COLOR training starts ...")
        device = self.device
        mesh = load_objs_as_meshes([target_path], device=device)
        
        # We scale normalize and center the target mesh to fit in a sphere of radius 1
        # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
        # to its original center and scale.  Note that normalizing the target mesh,
        # speeds up the optimization but is not necessary!
        verts = mesh.verts_packed()
        N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))
        
        # Get a batch of viewing angles.
        elev = torch.linspace(0, 360, num_views)
        azim = torch.linspace(-180, 180, num_views)
        
        # Place a point light in front of the object. As mentioned above, the front of
        # the cow is facing the -z direction.
        #lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])
        lights = AmbientLights(device=device)
        
        # Initialize an OpenGL perspective camera that represents a batch of different
        # viewing angles. All the cameras helper methods support mixed type inputs and
        # broadcasting. So we can view the camera from the a distance of dist=2.7, and
        # then specify elevation and azimuth angles for each viewpoint as tensors.
        R, T = look_at_view_transform(dist=2., elev=elev, azim=azim)
        cameras = PerspectiveCameras(device=device, R=R, T=T)
        
        # We arbitrarily choose one particular view that will be used to visualize
        # results
        camera = PerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...])
        
        
        # Define the settings for rasterization and shading. Here we set the output
        # image to be of size 128X128. As we are rendering images for visualization
        # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
        # rasterize_meshes.py for explanations of these parameters.  We also leave
        # bin_size and max_faces_per_bin to their default values of None, which sets
        # their values using heuristics and ensures that the faster coarse-to-fine
        # rasterization method is used.  Refer to docs/notes/renderer.md for an
        # explanation of the difference between naive and coarse-to-fine rasterization.
        raster_settings = RasterizationSettings(
            image_size=256*2, blur_radius=0.0, faces_per_pixel=1, perspective_correct=False
        )
        
        # Create a Phong renderer by composing a rasterizer and a shader. The textured
        # Phong shader will interpolate the texture uv coordinates for each vertex,
        # sample from a texture image and apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=camera, lights=lights),
        )
        
        # Create a batch of meshes by repeating the cow mesh and associated textures.
        # Meshes has a useful `extend` method which allows us do this very easily.
        # This also extends the textures.
        meshes = mesh.extend(num_views)
        
        # Render the cow mesh from each viewing angle
        target_images = renderer(meshes, cameras=cameras, lights=lights)
        
        del meshes
        # RGB images
        image_grid(target_images[...,:3].cpu().numpy(), rows=2, cols=5, rgb=True)
        plt.show()
    

        # create target cameras and target images
        target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
        target_cameras = [PerspectiveCameras(device=device, R=R[None, i, ...], 
                                           T=T[None, i, ...]) for i in range(num_views)]
        # realize cuda memory for training

        new_src_mesh = self.src_mesh.offset_verts(self.deform_verts)
         
        verts_shape = new_src_mesh.verts_packed().shape

        # train rgb color
        verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)

        # The optimizer
        # TODO optimization not good, large gitter during training, should penalize the invalid color
        optimizer = torch.optim.SGD([verts_rgb], lr=1, momentum=0.9)

        loop = tqdm(range(Niter))

        rgb_losses = []
        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()
            

            # penalize invalid color value
            norm_verts_rgb = torch.nn.functional.hardtanh(verts_rgb, min_val=0., max_val=1.)

            # Add per vertex colors to texture the mesh
            new_src_mesh.textures = TexturesVertex(verts_features=norm_verts_rgb) 
            
            # Randomly select two views to optimize over in this iteration.  Compared
            # to using just one view, this helps resolve ambiguities between updating
            # mesh shape vs. updating mesh texture
            loss = 0
            for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
                images_predicted = renderer(new_src_mesh, cameras=target_cameras[j], lights=lights)
        
                # Squared L2 distance between the predicted RGB image and the target 
                # image from our dataset
                predicted_rgb = images_predicted[...,:3]

                loss += ((predicted_rgb.squeeze() - target_rgb[j]) ** 2).mean()
                rgb_losses.append(float(loss.detach().cpu()))
            
            # Print the losses
            loss += ((norm_verts_rgb - verts_rgb)**2).sum()
            loop.set_description("rgb_loss = %.6f, vertice_loss:%.6f" % (loss,float(((norm_verts_rgb - verts_rgb)**2).mean().detach().cpu())))
            

            # Plot mesh
            if i % plot_period == 0:
                images_predicted = renderer(new_src_mesh, cameras=target_cameras[0], lights=lights)
                predicted_rgb = images_predicted[...,:3]

                plt.subplot(1,2,1); plt.imshow(predicted_rgb.squeeze().detach().cpu().numpy())
                plt.subplot(1,2,2); plt.imshow(target_rgb[0].detach().cpu().numpy())
                plt.show()

            # Optimization step
            loss.backward()
            optimizer.step()


        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        ax.plot(rgb_losses, label="color loss")
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Loss vs Color", fontsize="16");
        plt.show()

        # visualize all rendered images
        meshes = new_src_mesh.extend(num_views) 
        images = renderer(meshes, cameras=cameras, lights=lights)

        # RGB images
        image_grid(images[...,:3].detach().cpu().numpy(), rows=2, cols=5, rgb=True)
        plt.show()


        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

        # Scale normalize back to the original target size
        final_verts = final_verts * self.scale + self.offset

 
        final_verts = final_verts.detach().cpu().numpy()
        final_faces = final_faces.detach().cpu().numpy()
        final_verts_rgb = verts_rgb.detach().squeeze().cpu().numpy()

        tri_mesh = trimesh.Trimesh(vertices=final_verts, faces=final_faces, vertex_colors=final_verts_rgb)
        tri_mesh.export('no_color.obj')

    def load_and_scale_mesh(self, object_path):
        verts, faces, aux = load_obj(object_path)
        
        # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
        # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        # For this tutorial, normals and textures are ignored.
        faces_idx = faces.verts_idx.to(self.device)
        verts = verts.to(self.device)
        
        # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
        # (scale, center) will be used to bring the predicted mesh to its original center and scale
        # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        
        # We construct a Meshes structure for the target mesh
        return Meshes(verts=[verts], faces=[faces_idx]), center, scale


    
    
    def geometry_train(self, 
              Niter = 100, 
              w_chamfer = 1.0,
              w_edge = 1.0,
              w_normal = 0.01,
              w_laplacian = 0.1,
              plot_period = 4000):
    
        print("GEOMETRY training starts ...")
        # The optimizer
        optimizer = torch.optim.SGD([self.deform_verts], lr=1.0, momentum=0.9)
        
        loop = tqdm(range(Niter))
        
        # losses
        losses = {'chamfer_losses' : [],
                  'laplacian_losses' : [],
                  'edge_losses' : [],
                  'normal_losses' : [],}
        
        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()
        
            # Deform the mesh
            new_src_mesh = self.src_mesh.offset_verts(self.deform_verts)
        
            # We sample 5k points from the surface of each mesh 
            sample_trg = sample_points_from_meshes(self.trg_mesh, 1000)
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
            losses['chamfer_losses'].append(float(loss_chamfer.detach().cpu()))
            losses['edge_losses'].append(float(loss_edge.detach().cpu()))
            losses['normal_losses'].append(float(loss_normal.detach().cpu()))
            losses['laplacian_losses'].append(float(loss_laplacian.detach().cpu()))
        
            # Plot mesh
            if i % plot_period == 0:
                plot_pointcloud(new_src_mesh, i)
        
                # save intermediate mesh
                if False: 
                    with torch.no_grad():
                        # Fetch the verts and faces of the final predicted mesh
                        verts, faces = new_src_mesh.get_mesh_verts_faces(0)
        
                        # Scale normalize back to the original target size
                        rescaled_verts = verts * self.scale + self.offset
        
                        # Store the predicted mesh using save_obj
                        obj_path = os.path.join('./morphy_results', 'iter_%05d.obj'%i)
                        save_obj(obj_path, rescaled_verts, faces)
            # Optimization step
            loss.backward()
            optimizer.step()
    
    
        # visualize the losses
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        ax.plot(losses['chamfer_losses'], label="chamfer loss")
        ax.plot(losses['edge_losses'], label="edge loss")
        ax.plot(losses['normal_losses'], label="normal loss")
        ax.plot(losses['laplacian_losses'], label="laplacian loss")
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Loss vs iterations", fontsize="16");
        #plt.show()

        # save results
        # Fetch the verts and faces of the final predicted mesh
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        
        # Scale normalize back to the original target size
        final_verts = final_verts * self.scale + self.offset
        
        # Store the predicted mesh using save_obj
        final_obj = os.path.join('./', 'geometry_result.obj')
        save_obj(final_obj, final_verts, final_faces)



TheCreator()
