import blenderproc as bproc
import json
import argparse
import os

# try fetch object learning workspace in global environmemt
try:
    OBJECT_LEARNING_WS = os.environ["OBJECT_LEARNING_WORKSPACE_PATH"]
except KeyError as e:
    print("WARNING: OBJECT_LEARNING_WORKSPACE has not been initialized in env. Please source object_learning.env")
    OBJECT_LEARNING_WS = '' 

import glob
import numpy as np
import random
import time

class COCODataGenerator:
    def __init__(self, args):
        self.args = args

        self.target_bop_objs = []
        self.distract_bop_objs = []

        # load object in bproc
        self.load_target_objects()
        self.load_distractor_objects()

        # define coco file
        sf = "coco_data" if self._dataset_name is None else self._dataset_name

        subDir = args.sub_dir
        self._coco_dir = os.path.join(self.args.output_dir, sf,  subDir)

        # load cc_textureccs for room background
        self.cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

        # load other textures for object (optional), used assets, please view all material in cc
        self.obj_textures = bproc.loader.load_ccmaterials(
            args.cc_textures_path,
            used_assets=["Wood001", "Wood002", "Wood017", "Wood022", "Wood046", "Wood047"],
        )

        # uniform color array
        self.uni_colors = np.array([
                                    [0, 0, 255],  # blue
                                    [0, 255, 0],  # green
                                    [255, 0, 0],  # red
                                    [255, 255, 0],
                                   ])  # yellow


        # load camera setting
        self.load_camera_setting()

        # set up environment and render
        self.main()

        # reformatting for detectron2 training
        self.reformat_coco_anns()

    def reformat_coco_anns(self):
        #TODO remove distract object IDs
        coco_fp = os.path.join(self._coco_dir, "coco_annotations.json")

        if not os.path.exists(coco_fp):
            print(f"WARNING: COCO annotation file({coco_fp}) not exists")
            return

        # load and modify category, overwrite the category in coco annotation with definition in instances.json
        json_fpath = os.path.join(self.args.object_models_path, "instances.json")

        with open(json_fpath, 'r') as f:   
            instances = json.load(f)       

        categories = instances['categories']

        with open(coco_fp, 'r') as f:
            cocoAnns = json.load(f)

        cocoAnns['categories'] = categories

        with open(coco_fp, 'w') as f:
            json.dump(cocoAnns, f)
        
    # Define a function that samples 6-DoF poses
    @staticmethod
    def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
        max = np.random.uniform([0.2, 0.2, 0.6], [0.3, 0.3, 0.8])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
   
    def load_target_objects(self):
        model_path = self.args.object_models_path
        json_fpath = os.path.join(model_path, "instances.json")
        num_each_object = self.args.num_of_each_object

        if not os.path.isfile(json_fpath):
            raise FileNotFoundError(f"ERROR: {json_fpath} not found.")

        with open(json_fpath, 'r') as f:
            instances = json.load(f)
                                   
        datasetName = instances['dataset_name']
        self._dataset_name = datasetName
        objectLists = instances['categories']  # object info dict contains "supercategory", "id", "filename", "name"

        print("INFO: Objects are being loaded ...")
        # load target objects      
        for obj in objectLists:    
            objId = obj["id"]      
            objName = obj["name"]
            objSupercategoty = obj["supercategory"]
            # duplicate the same objects, maximum number
            for _ in range(num_each_object):
                obj_path = os.path.join(model_path,  obj["filename"])

                if not os.path.exists(obj_path):
                    raise FileNotFoundError(f"ERROR: object file not found.")
                                   
                self.target_bop_objs += bproc.loader.load_obj(obj_path)
                                   
                # rescale 3D model from mm to m
                if self.args.mm2m:
                    self.target_bop_objs[-1].set_scale([0.001, 0.001, 0.001])
                                   
                self.target_bop_objs[-1].set_cp("category_id", objId)
                                   
                # optional         
                self.target_bop_objs[-1].set_cp("class_name", objName)
                self.target_bop_objs[-1].set_cp("supercategory", objSupercategoty)
                if datasetName is not None:
                    self.target_bop_objs[-1].set_cp("custom_dataset_name", datasetName)

            print(
                f"INFO: {obj['name']} loaded, {len(self.target_bop_objs)} target objects totally loaded in blender buffer"
            )

    def load_distractor_objects(self):
        distractor_datasets = self.args.distractor_datasets
        for distractor in distractor_datasets:
            if distractor == "tless":
                self.distract_bop_objs += bproc.loader.load_bop_objs(
                    bop_dataset_path=os.path.join(self.args.bop_parent_path, distractor),
                    model_type="cad",
                    mm2m=True,
                )
            else:
                self.distract_bop_objs += bproc.loader.load_bop_objs(
                    bop_dataset_path=os.path.join(self.args.bop_parent_path, distractor), mm2m=True
                )

    def load_camera_setting(self):
        # load BOP datset intrinsics, use ycbv camera setting in current
        bproc.loader.load_bop_intrinsics(
            bop_dataset_path=os.path.join(self.args.bop_parent_path, "ycbv")
        )

    def main(self):
        print("INFO: Main thread starts")
        target_obj_num = self.args.target_obj_num
        distractor_obj_num = self.args.distractor_obj_num
        cam_view_num = self.args.cam_view_num
        target_object_render_mode = self.args.target_object_render_mode

        # set up environment
        # set shading and hide objects
        for obj in self.target_bop_objs + self.distract_bop_objs:
            obj.set_shading_mode("auto")
            obj.hide(True)

        # create room with side of 2 m
        room_planes = [
            bproc.object.create_primitive("PLANE", scale=[2, 2, 1]),
            bproc.object.create_primitive(
                "PLANE", scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]
            ),
            bproc.object.create_primitive(
                "PLANE", scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]
            ),
            bproc.object.create_primitive(
                "PLANE", scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]
            ),
            bproc.object.create_primitive(
                "PLANE", scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0]
            ),
        ]
        for plane in room_planes:
            plane.enable_rigidbody(
                False,
                collision_shape="BOX",
                mass=1.0,
                friction=100.0,
                linear_damping=0.99,
                angular_damping=0.99,
            )


        # sample light color and strenght from ceiling
        light_plane = bproc.object.create_primitive(
            "PLANE", scale=[3, 3, 1], location=[0, 0, 10]
        )
        light_plane.set_name("light_plane")
        light_plane_material = bproc.material.create("light_material")

        # sample point light on shell
        light_point = bproc.types.Light()
        light_point.set_energy(300)


        # activate depth rendering without antialiasing and set amount of samples for color rendering
        bproc.renderer.enable_depth_output(activate_antialiasing=True)
        bproc.renderer.set_max_amount_of_samples(50)

        # render
        st = time.time()
        for i in range(self.args.num_scenes):
            # Sample bop objects for a scene
            # NOTE numbers of target and distract object in the scene
            sampled_target_bop_objs = list(
                np.random.choice(self.target_bop_objs, size=target_obj_num, replace=False)
            )
            
            if not len(self.distract_bop_objs) == 0:
                sampled_distractor_bop_objs = list(
                    np.random.choice(self.distract_bop_objs, size=distractor_obj_num, replace=False)
                )
            else:
                sampled_distractor_bop_objs = []

            # Randomize materials and set physics
            # for obj in (sampled_distractor_bop_objs):
            for obj in sampled_target_bop_objs + sampled_distractor_bop_objs:
                mat = obj.get_materials()[0]

                if "custom_dataset_name" in obj.get_all_cps():
                    if obj.get_cp("custom_dataset_name") in ["toyblock"]:
                        # only toyblocks for diverse wooden textures
                        texture = np.random.choice(self.obj_textures)
                        obj.replace_materials(texture)

                    if target_object_render_mode == 1:
                        # vertice color
                        mat.map_vertex_color()

                    if target_object_render_mode == 2:
                       # RGB must be a float 0~1
                       rgb = np.random.uniform(0.1, 0.9, size=3)
                       #obj.clear_materials()                                #optional
                       rgba = rgb.tolist() + [1]
                       mat.set_principled_shader_value("Base Color", rgba)


                # BOP dataset objects
                if "bop_dataset_name" in obj.get_all_cps():
                    if obj.get_cp("bop_dataset_name") in ["tless", "itodd"]:
                        grey_col = np.random.uniform(0.1, 0.9)
                        #obj.clear_materials()                                #optional
                        mat.set_principled_shader_value(
                            "Base Color", [grey_col, grey_col, grey_col, 1]
                        )

                                                                                                  
                mat.set_principled_shader_value("Roughness", np.random.uniform(0.5, 1))           # 0 for smooth, and reflective
                mat.set_principled_shader_value("Specular", np.random.uniform(0.5, 1))            # refers to material

                obj.enable_rigidbody(
                    True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99
                )
                obj.hide(False)

            # Sample two light sources
            light_plane_material.make_emissive(
                emission_strength=np.random.uniform(3, 6),
                emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
            )
            light_plane.replace_materials(light_plane_material)
            light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))

            location = bproc.sampler.shell(
                center=[0, 0, 0],
                radius_min = self.args.light_radius_min,  
                radius_max = self.args.light_radius_max,
                elevation_min = self.args.light_elevation_min,
                elevation_max = self.args.light_elevation_max,
            )
            light_point.set_location(location)

            # sample CC Texture and assign to room planes
            random_cc_texture = np.random.choice(self.cc_textures)
            for plane in room_planes:
                plane.replace_materials(random_cc_texture)

            # Sample object poses and check collisions
            bproc.object.sample_poses(
                objects_to_sample=sampled_target_bop_objs + sampled_distractor_bop_objs,
                sample_pose_func=self.sample_pose_func,
                max_tries=1000,
            )

            # Physics Positioning
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=3,
                max_simulation_time=10,
                check_object_interval=1,
                substeps_per_frame=20,
                solver_iters=25,
            )

            # BVH tree used for camera obstacle checks
            bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(
                sampled_target_bop_objs + sampled_distractor_bop_objs
            )

            cam_poses = 0

            # numbers of camera for one scene
            while cam_poses < cam_view_num:
                # Sample location
                # NOTE radius, how far camera away from interest area
                location = bproc.sampler.shell(
                    center=[0, 0, 0],
                    radius_min = self.args.camera_radius_min,  
                    radius_max = self.args.camera_radius_max,
                    elevation_min = self.args.camera_elevation_min,
                    elevation_max = self.args.camera_elevation_max,
                )
                # Determine point of interest in scene as the object closest to the mean of a subset of objects
                interest_object_num = int((target_obj_num + distractor_obj_num) * self.args.camera_magnification_factor)
                poi = bproc.object.compute_poi(
                    np.random.choice(sampled_target_bop_objs, size=interest_object_num, replace=False)
                )
                # Compute rotation based on vector going from location towards poi
                rotation_matrix = bproc.camera.rotation_from_forward_vec(
                    poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159)
                )
                # Add homog cam pose based on location an rotation
                cam2world_matrix = bproc.math.build_transformation_mat(
                    location, rotation_matrix
                )

                # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
                if bproc.camera.perform_obstacle_in_view_check(
                    cam2world_matrix, {"min": 0.3}, bop_bvh_tree
                ):
                    # Persist camera pose
                    bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                    cam_poses += 1

            # # activate normal rendering
            bproc.renderer.enable_normals_output()

            # Render segmentation data and produce instance attribute maps
            bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={'category_id': 0})

            # render the whole pipeline
            data = bproc.renderer.render()

            # Write data to coco file
            bproc.writer.write_coco_annotations(self._coco_dir,
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    mask_encoding_format="polygon",                 # or 'rle'
                                    colors=data["colors"],
                                    color_file_format="JPEG")

            # Write data in bop format for bop datasets challenage, e.g 6D pose, depth
            #bproc.writer.write_bop(
            #    os.path.join(self.args.output_dir, "bop_data"),
            #    sampled_target_bop_objs,
            #    depth_scale=0.1,
            #    depths=data["depth"],
            #    colors=data["colors"],
            #    color_file_format="JPEG",
            #    m2mm=True,
            #)

            # reset
            for obj in sampled_target_bop_objs + sampled_distractor_bop_objs:
                obj.disable_rigidbody()
                obj.hide(True)

            # time evaluation 
            et = time.time()
            aver_t = (et - st) / (i + 1)

            rt = aver_t * (args.num_scenes - i - 1)
            rh = rt // 60 // 60
            rm = (rt - rh * 3600) // 60
            noe = int((i + 1) / args.num_scenes * 50)
            fs = f"INFO: Blenderproc in status: [{'='*noe}{' '*(50-noe)}] {(i+1)/args.num_scenes*100:.2f}% done. Rendering 25 rgbd images takes {aver_t} s on average. it will end in {rh} hours {rm} mins"
            print(fs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_models_path",
        default= os.path.join(OBJECT_LEARNING_WS, "data/models"),
        help="Path of target object models",
    )
    parser.add_argument(
        "--distractor_datasets",
        default=['ycbv'],
        help="Subfolder of bop dataset for distractor objects",
    )
    parser.add_argument(
        "--bop_parent_path",
        default="/hri/localdisk2/datasets/bop",
        help="Path to the bop datasets parent directory",
    )
    parser.add_argument(
        "--cc_textures_path",
        default="/hri/localdisk2/blenderRef/cctextures",
        help="Path to downloaded cc textures",
    )
    parser.add_argument(
        "--output_dir",
        default= os.path.join(OBJECT_LEARNING_WS, "outputs/datasets"),
        help="Path to where the final files will be saved ",
    )
    parser.add_argument(
        "--sub_dir",
        default= "bproc_train",
        help="Sub directory under output dir, e.g., bproc_train, bproc_val, bproc_test",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=2,
        help="How many scenes with 25 images(default) each to generate",
    )
    parser.add_argument(
        "--num_of_each_object",                       # maximum
        type=int,
        default=2,
        help="Frequency of each object's appearance",
    )
    parser.add_argument(
        "--target_object_render_mode",
        type=int,
        default=0,
        help="Render mode ",
    )
    parser.add_argument(
        "--mm2m",
        default=True,
        help="Rescale object size from mm to m",
    )
    parser.add_argument(
        "--target_obj_num",
        type=int,
        default=1,
        help="Number of target object in a scene",
    )
    parser.add_argument(
        "--distractor_obj_num",
        type=int,
        default=6,
        help="Number of target object in a scene",
    )
    parser.add_argument(
        "--cam_view_num",
        type=int,
        default=25,
        help="Number of camera views in a scene",
    )
    parser.add_argument(
        "--camera_radius_min",
        type=float,
        default=0.5,
        help="Camera minimum radius",
    )
    parser.add_argument(
        "--camera_radius_max",
        type=float,
        default=0.7,
        help="Camera maximum radius",
    )
    parser.add_argument(
        "--camera_elevation_min",
        type=int,
        default=30,
        help="Camera minimum elevation",
    )
    parser.add_argument(
        "--camera_elevation_max",
        type=int,
        default=89,
        help="Camera minimum elevation",
    )
    parser.add_argument(
        "--light_radius_min",
        type=float,
        default=1.5,
        help="Light minimum radius",
    )
    parser.add_argument(
        "--light_radius_max",
        type=float,
        default=2.5,
        help="Light maximum radius",
    )
    parser.add_argument(
        "--light_elevation_min",
        type=int,
        default=10,
        help="Light minimum elevation",
    )
    parser.add_argument(
        "--light_elevation_max",
        type=int,
        default=89,
        help="Light minimum elevation",
    )
    parser.add_argument(
        "--camera_magnification_factor",
        type=float,
        default=0.8,
        help="A magnification factor to determine point of interest in scene as the object closest to the mean of a subset of objects (range is from 0 to 1, with smaller values indicating that the camera focuses more on small portions of the object.)",
    )
    args = parser.parse_args()

    # initialize the scene
    bproc.init()

    cdg = COCODataGenerator(args)


