import copy

import numpy as np
import open3d
import pyrender


class VisPyrender:
    def __init__(self, width=1280, height=720, scale_factor=1, light=False):

        if light:
            self.__scene = pyrender.Scene(ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))
        else:
            self.__scene = pyrender.Scene()
        self.__cindex = scale_factor
        self.__width = width // scale_factor
        self.__height = height // scale_factor
        self.__camera = None
        self.__r = pyrender.OffscreenRenderer(viewport_width=self.__width, viewport_height=self.__height)

    def set_IntrinsicsCamera(self, intrinsic_matrix):
        if self.__camera is None:
            intrinsic = copy.deepcopy(intrinsic_matrix)
            fx = intrinsic[0][0] / self.__cindex
            fy = intrinsic[1][1] / self.__cindex
            cx = intrinsic[0][2] / self.__cindex
            cy = intrinsic[1][2] / self.__cindex
            self.__camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy)

    def get_view_size(self):
        return self.__height, self.__width

    def add_geometry(self, mesh, object_pose=np.eye(4, dtype="float32")):
        # mesh should be pyrender.Mesh, object_pose: ((4,4),float32)
        mesh_copy = copy.deepcopy(mesh)
        return self.__scene.add(mesh_copy, pose=object_pose.copy())

    def update_view_point(self, extrinsic):
        # add a camera in scene with camera_pose
        camera_pose = np.linalg.inv(extrinsic.copy())
        camera_pose[:3, 1:3] = -camera_pose[:3, 1:3]
        self.__scene.add(self.__camera, pose=camera_pose)

    def capture_depth_float_buffer(self):
        # capture rendered depth image, rendered depth image in meters
        return self.__r.render(self.__scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)

    def capture_float_buffer(self):
        # capture rendered depth image
        return self.__r.render(self.__scene)

    def clear_geometries(self):
        # clear geometries and camera in scene
        self.__scene.clear()

    def quick_depth_render(self, mesh, object_pose, extrinsic):
        self.clear_geometries()
        self.add_geometry(mesh, object_pose)
        self.update_view_point(extrinsic)
        return self.capture_depth_float_buffer()

    def optimization_render(self, mesh, object_pose, extrinsic):
        return self.quick_depth_render(mesh, object_pose, extrinsic)


class VisOpen3D:
    def __init__(self, width=1280, height=720):

        # launch a window, and render
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window(width=width, height=height, visible=False)
        self.__width = width
        self.__height = height

    def clear_geometries(self):
        self.__vis.clear_geometries()

    def add_geometry(self, mesh, pose):
        mesh.transform(pose)
        self.__vis.add_geometry(mesh, reset_bounding_box=True)

    def capture_depth_float_buffer(self):
        return self.__vis.capture_depth_float_buffer(do_render=True)

    def update_view_point(self, intrinsic, extrinsic):
        ctr = self.__vis.get_view_control()
        param = self.convert_to_open3d_param(intrinsic, extrinsic)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()

    def convert_to_open3d_param(self, intrinsic, extrinsic):
        param = open3d.camera.PinholeCameraParameters()
        param.intrinsic = open3d.camera.PinholeCameraIntrinsic()
        # use set_intrinsics function
        height = self.__height
        width = self.__width
        fx = intrinsic[0][0]
        cx = width / 2 - 0.5
        fy = intrinsic[1][1]
        cy = height / 2 - 0.5
        param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        param.extrinsic = extrinsic
        return param
