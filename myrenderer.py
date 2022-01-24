import sys
import torch
import pytorch3d

"""
#If imports fail:
if need_pytorch3d:
    if torch.__version__.startswith("1.9") and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        version_str="".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".",""),
            f"_pyt{torch.__version__[0:5:2]}"
        ])
        !pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
    else:
        # We try to install PyTorch3D from source.
        !curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
        !tar xzf 1.10.0.tar.gz
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        !pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
"""
import time
import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import scipy.optimize

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate, axis_angle_to_matrix

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

class Renderer(nn.Module):
    #def __init__(self, obj_fname="./data/teapot.obj", image_size=(180,320)):
    #def __init__(self, obj_fname="./data/teapot.obj", image_size=(180,320), ref_pos = [2.6, .1, 2.4], init_pos=[3.50,  .1, 2.5]):
    @torch.no_grad()
    def __init__(self, obj_fname="./data/teapot.obj", image_size=(300,300), ref_pos = [0, 1., 2.11, 0., .0, .0], init_pos=[0.0, 0.0, 2.11, 0.2,  .3, .6]):
        obj_fname = '/hri/localdisk/franzius/intro_ros_ws/src/intro_object_models/models/ycb-video/006_mustard_bottle/textured_simple.obj'
        super(Renderer, self).__init__()
        self.init_pos = init_pos
        self.ref_pos = ref_pos
        # Set the cuda device 
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
            print ("Using cuda device", self.device)
        else:
            self.device = torch.device("cpu")
            print ("using cpu")

        self.image_size = image_size
        # Load the obj and ignore the textures and materials.
        verts, faces_idx, _ = load_obj(obj_fname)
        verts[:] *=20  # increase model size
        faces = faces_idx.verts_idx

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        self.meshes = Meshes(verts=[verts.to(self.device)],faces=[faces.to(self.device)], textures=textures)

        # Initialize a perspective camera.
        self.cameras = PerspectiveCameras(focal_length=.2, device=self.device)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        raster_settings = RasterizationSettings(image_size=self.image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, faces_per_pixel=1)
        self.lights = PointLights(device=self.device, location=((2.0, 2.0, -2.0),))
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.set_ref_image(ref_pos)
        self.set_camera_position(init_pos)

    def set_camera_position(self, arr):
         self.camera_position = nn.Parameter(torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.meshes.device))       

    def render_tensor(self, get_image=False):
        #R = look_at_rotation(self.camera_position[None, 3:], device=self.device)  # (1, 3, 3)
        R = axis_angle_to_matrix(self.camera_position[3:])
        R = R.reshape([1,3,3])
        #T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :3, None])[:, :, 0]   # (1, 3)
        T = self.camera_position[:3]
        T = T.reshape([1,3])
        fragments = self.rasterizer(self.meshes.clone(),  R=R, T=T)
        if get_image:
            images = self.shader(fragments, self.meshes.clone(), R=R, T=T)
            return fragments.zbuf, images
        else:
            return fragments.zbuf

    def render_image(self, get_image=False):
        if get_image:
            zbuf, images = self.render_tensor(get_image=True)
            zbuf = zbuf.detach().cpu().numpy().squeeze()
            images = images.detach().cpu().numpy().squeeze()
            return zbuf, images
        else:
            zbuf = self.render_tensor(get_image=False)
            zbuf = zbuf.cpu().numpy().squeeze()
            return zbuf

    @torch.no_grad()
    def set_ref_image(self, ref_pose):
        self.ref_pose = ref_pose
        self.set_camera_position(ref_pose)
        self.ref_z_tensor = self.render_tensor(get_image=False)
        self.ref_z_image, self.ref_image = self.render_image(get_image=True)

    def forward(self):
        # Create an optimizable parameter for the x, y, z position of the camera. 
        zbuf = self.render_tensor(get_image=False)
        #loss = torch.sum(torch.abs(zbuf - self.ref_z_tensor) )
        #loss = torch.clip((zbuf - self.ref_z_tensor)**2, 0, 0.01)
        #loss = torch.sum(loss)
        maskA = zbuf>0
        maskB = self.ref_z_tensor>0
        depthDiff = zbuf - self.ref_z_tensor
        #depthDiff[torch.logical_not(maskB)] /= 100
        #depthDiff = depthDiff[maskB]
        
        loss = torch.clip(torch.abs(depthDiff), 0, 0.1).sum()
        loss = torch.abs(depthDiff).sum()
        
        if True:
            overlap = (torch.bitwise_and(maskA, maskB)*1.).sum()/maskB.sum()
            overlap = torch.autograd.Variable(overlap, requires_grad=True)
            #print (float(loss), float(overlap))
            loss = 1000 * overlap  + loss

        #print("loss: %.2f" %(loss.data), "current pos:", self.camera_position.clone().detach().cpu().numpy(), "ref pos:", self.ref_pose)
        return loss#, zbuf.detach().cpu().numpy().squeeze()

    @torch.no_grad()
    def estimate_gradient(self, eps=0.001):
        tic = time.time()
        grad = torch.zeros_like(self.camera_position)
        loss0 = self.forward()
        for i in range(len(grad)):
            self.camera_position[i]+=eps
            loss = self.forward()
            self.camera_position[i]-=eps
            grad[i] = (loss0-loss)/eps
        toc = time.time()
        print("grad dt: %.6fs" %(toc-tic), grad, "at point:", self.camera_position)
        return grad, loss0

    def callback(self, params):
        self.set_camera_position(params)
        return self().detach().cpu().numpy()

def optimize(visualize=True):
    model = Renderer()

    initial_zBuf, initial_image = model.render_image(get_image=True)
    initial_zBuf = initial_zBuf.copy()
    initial_image = initial_image.copy()

    optimizer = torch.optim.Adam([model.camera_position], lr=0.0002)
    #optimizer = torch.optim.Rprop(model.parameters(), lr=0.0001)

    filename_output = "./teapot_optimization_demo.gif"
    N_steps = 10
    vis_iter = 1
    loop = tqdm(range(N_steps))
    losses = np.zeros(N_steps)
    params = np.zeros([N_steps, 6])
    if visualize:
        writer = imageio.get_writer(filename_output, mode='I', duration=0.03)
        X = (int((N_steps//vis_iter)**0.5)); Y=int(np.ceil((N_steps/vis_iter)/X))
        cnt = 1

    tic = time.time()
    eps = 0.005
    for i in loop:
        #if i==50: eps = 0.01
        optimizer.zero_grad()
        #loss = model()
        #loss.backward()#retain_graph=True)
        grad,loss = model.estimate_gradient()
        #optimizer.step() 
        with torch.no_grad():
            #print ("A", grad)
            grad = grad/((grad**2).sum()**0.5)
            grad[:3] *= 0.02
            #print("B", grad)
            model.camera_position += eps*grad
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        losses[i] = float(loss) #float(loss.data)
        print (i, float(loss))
        params[i,:] = model.camera_position.detach().cpu().numpy()
        #if loss.item() < 200:
        #    break
    
        # Save outputs to create a GIF. 
        if visualize and i % vis_iter == 0:
            zbuf,img = model.render_image(get_image=True)        
            tmp = np.clip(zbuf.copy(), 0, 5) / 5.
            tmp = (tmp*255).astype('uint8')
            plt.subplot(X,Y,cnt); plt.imshow(tmp);
            writer.append_data(tmp)
            cnt+=1
            plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
            plt.axis("off")

    print ("dt:", time.time()-tic)
    if visualize:
        plt.figure()
        zBuf, image = model.render_image(get_image=True)
        plt.subplot(4,2,1); plt.imshow(model.ref_image); plt.title('ref image')
        plt.subplot(4,2,2); plt.imshow(model.ref_z_image); plt.title('ref depth image')
        plt.subplot(4,2,3); plt.imshow(initial_image); plt.title('initial image')
        plt.subplot(4,2,4); plt.imshow(initial_zBuf); plt.title('initial depth image')

        plt.subplot(4,2,5); plt.imshow(image); plt.title('optimized image')
        plt.subplot(4,2,6); plt.imshow(zBuf); plt.title('optimized depth image')
        plt.subplot(4,2,7); plt.imshow(zBuf-model.ref_z_image); plt.title('diff ref depth to optimized epth'); plt.colorbar()
        plt.subplot(4,2,8); plt.plot(losses); plt.title('loss')
        plt.figure(); plt.subplot(1,2,1); plt.title('param evolution: T')
        colors = 'bgr'
        for i in range(3):
            plt.plot([model.ref_pos[i]]*len(losses), '.%s'%(colors[i]))
            plt.plot(params[:,i], '%s'%(colors[i]))

        plt.subplot(1,2,2); plt.title('param evolution: R')
        for i in range(3):
            plt.plot([model.ref_pos[i+3]]*len(losses), '.%s'%(colors[i]))
            plt.plot(params[:,i+3], '%s'%(colors[i]))
        
        

    
    if visualize: writer.close()
    plt.show()
    return model.camera_position.detach().cpu().numpy()

if __name__ == "__main__":
    optimize()
    1/0
    #with torch.no_grad():
    import pylab
    ren =  Renderer()
    N= 100
    zbuf,img = ren.render_image(get_image=True)
    tic = time.time()
    for i in range(N):
        ren.render_tensor(get_image=True)
    tic2 = time.time()
    for i in range(N):
       ren.render_tensor(get_image=False)
    dt1 = time.time()-tic2
    dt2 = tic2-tic

    tic = time.time()
    for i in range(N):
        ren.render_image(get_image=True)
    tic2 = time.time()
    for i in range(N):
       ren.render_image(get_image=False)
    dt3 = time.time()-tic2
    dt4 = tic2-tic

    print("resolution:", ren.image_size)
    print("image dt: %.6fs, only depth: %.6fs" %(dt2/N, dt1/N))
    print("tensor dt: %.6fs, only depth: %.6fs" %(dt4/N, dt3/N))

    pylab.subplot(1,2,1)
    pylab.imshow(zbuf); pylab.colorbar()
    pylab.subplot(1,2,2)
    pylab.imshow(img)
    pylab.show()
    

