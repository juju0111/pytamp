import numpy as np
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pytamp.benchmark.benchmark import Benchmark

from pytamp.utils.making_scene_utils import Make_Scene
from pytamp.scene.scene_manager import SceneManager

import trimesh 
import easydict
from copy import deepcopy

from pytamp.utils.making_scene_utils import load_mesh, get_obj_name, Make_Scene
from pykin.utils.mesh_utils import get_object_mesh
from pykin.utils import plot_utils as p_utils
from pykin.utils import mesh_utils as m_utils


from pykin.utils.kin_utils import ShellColors as sc

class Rearrange1(Benchmark):
    def __init__(
        self,
        robot_name="panda",
        object_names=None,
        init_scene: Make_Scene = None,
        goal_scene: Make_Scene = None,
        geom="visual",
        is_pyplot=True,
    ):
        """

        Args :
            robot_name (str) : robot name to use
            object_names : object name list on the support object
            init_scene (Scene for Acronym)
            goal_scene (Scene for Acronym)
        """

        self.object_names = object_names
        self.param = {
            "object_names": self.object_names,
            "goal_scene": goal_scene._poses,
        }
        self.benchmark_config = {0: self.param}
        self.init_scene = init_scene
        self.obj_colors = None
        super().__init__(robot_name, geom, is_pyplot, self.benchmark_config)

        self._load_robot()
        self._load_objects()
        self._load_scene(self.init_scene, self.scene_mngr)

        self.scene_mngr.scene._init_bench_rearrange()

        # 처음 scene_mngr 생성 시, scene.mngr.scene에서 goal 조건 셋팅해서 따로 필요없음
        # 단지 시각화 위해서만 잠깐 쓰임. 없어도 됨.
        # define goal_scene
        self.goal_scene = goal_scene
        self.goal_scene_mngr = SceneManager(
            self.geom,
            is_pyplot=self.is_pyplot,
            benchmark=self.benchmark_config,
            debug_mode=True,
        )
        self._load_scene(self.goal_scene, self.goal_scene_mngr)

    def _load_robot(self):
        self.robot = SingleArm(
            f_name=self.urdf_file,
            offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]),
            has_gripper=True,
            gripper_name=self.gripper_name,
        )
        if self.robot_name == "panda":
            self.robot.setup_link_name("panda_link_0", "right_hand")
            self.robot.init_qpos = np.array(
                [
                    0,
                    np.pi / 16.0,
                    0.00,
                    -np.pi / 2.0 - np.pi / 3.0,
                    0.00,
                    np.pi - 0.2,
                    -np.pi / 4,
                ]
            )
        if self.robot_name == "doosan":
            self.robot.setup_link_name("base_0", "right_hand")
            self.robot.init_qpos = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0])

    def _load_objects(self):
        """
        Re-transform the scene created with acronym_tool
            to fit the pytamp scene with the robot on it
        """

        self.table_mesh = self.init_scene._support_objects["table"]

        # Set table_pose according to robot height
        self.table_pose = Transform(pos=np.array([0.9, -0.6, 0.043]))
        self.scene_mngr.table_pose = self.table_pose

        # object_pose transformation according to Table Pose
        for o_name, o_pose in self.init_scene._poses.items():
            self.init_scene._poses[o_name] = self.table_pose.h_mat @ o_pose

        for o_name, o_pose in self.param["goal_scene"].items():
            self.param["goal_scene"][o_name] = self.table_pose.h_mat @ o_pose

        # assign color
        self.init_scene.colorize()

    def _load_scene(self, scene: Make_Scene, scene_mngr: SceneManager):
        """
        Args:
            scene (Make_Scene) : Random scene with object mesh and pose created with Acronym_tool
            scene_mngr (SceneManager) : scene manager for creating specific initial and goal scenes
        """
        scene_mngr.add_object(
            name="table",
            gtype="mesh",
            gparam=self.table_mesh,
            h_mat=scene._poses["table"],
            color=[0.823, 0.71, 0.55],
        )

        logical_states = [(f"{o_name}", ("on", "table")) for o_name in self.object_names]

        if not self.obj_colors:
            self.obj_colors = [
                np.array(scene._objects[o_name].visual.face_colors[:, :3][0])
                for o_name in self.object_names
            ]

        # set object, logical_state in scene_mngr
        for i, o_name in enumerate(self.object_names):
            scene_mngr.add_object(
                name=o_name,
                gtype="mesh",
                gparam=scene._objects[o_name],
                h_mat=scene._poses[o_name],
                color=self.obj_colors[i],
            )
            scene_mngr.set_logical_state(logical_states[i][0], logical_states[i][1])

        scene_mngr.add_robot(self.robot, self.robot.init_qpos)
        scene_mngr.set_logical_state("table", (scene_mngr.scene.logical_state.static, True))
        scene_mngr.set_logical_state(
            scene_mngr.gripper_name, (scene_mngr.scene.logical_state.holding, None)
        )
        scene_mngr.update_logical_states(is_init=True)
        scene_mngr.show_logical_states()

    def render_axis(self, scene_mngr: SceneManager):
        for o_name, _ in self.scene_mngr.scene.objs.items():
            pose = scene_mngr.scene.objs[o_name].h_mat
            scene_mngr.render.render_axis(pose)

    def get_obj_point_clouds(self, scene, manipulate_obj_name):
        point_clouds = trimesh.PointCloud(np.zeros((1,3))).vertices
        pc_count = 0
        pass_count = 0
        for _, item in enumerate(scene.objs.items()):
            # The support object only considers the top plane
            # So there is no need for point clouds in other parts of the mesh except for the top plane.
            name, i = item

            if manipulate_obj_name == name:
                pc_count = _

            if name in self.init_scene._support_objects.keys():
                pass_count += 1
                continue

            copied_mesh = deepcopy(i.gparam)
            copied_mesh.apply_translation(-i.gparam.center_mass)
            copied_mesh.apply_transform(i.h_mat)

            # random sampling으로 mesh 위 point cloud 일부 가져오기 
            points = copied_mesh.sample(1000)
            point_clouds = np.vstack([point_clouds, points])
            # print(name, len(point_clouds))

        return point_clouds[1:], pc_count-pass_count
    
    def get_support_space_point_cloud(self):
        # consider table top point cloud..! 
        # In the case of drawers and bookshelves, it is a little lacking to consider
        support_polys, support_T, sup_obj_name = self.init_scene._get_support_polygons()

        support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]

        pts = trimesh.path.polygons.sample(
                            support_polys[support_index], count=3000
                        )
        z_arr = np.full((len(pts), 1), 0)
        o_arr = np.full((len(pts), 1), 1)

        sup_point_cloud = np.hstack((pts, z_arr))
        sup_point_cloud = np.hstack((sup_point_cloud, o_arr))

        transformed_point_cloud = np.dot(support_T[support_index][:3], 
                                         sup_point_cloud.T).T + self.scene_mngr.scene.objs[sup_obj_name[support_index]].h_mat[:3,3]

        return transformed_point_cloud
    
    def get_combined_point_cloud(self, current_scene, next_scene, obj_name, c_pc, n_pc,count,):
        # Transformation matrix from current_scene obj_pose to next_scene obj_pose
        cTn = m_utils.get_relative_transform(current_scene.objs[obj_name].h_mat,
                                             next_scene.objs[obj_name].h_mat)
        
        o_arr = np.full((len(c_pc), 1), 1)
        c_pc_h = np.hstack((c_pc, o_arr))

        transformed_pc =  np.dot(c_pc_h, cTn[:3].T)
        
        # obj mean x,y position
        current_mean = np.array([np.mean(transformed_pc[1000*count:1000*(count+1)][:,0]), 
                                np.mean(transformed_pc[1000*count:1000*(count+1)][:,1]),
                                np.mean(transformed_pc[1000*count:1000*(count+1)][:,2])])
        next_mean = np.array([np.mean(n_pc[1000*count:1000*(count+1)][:,0]), 
                              np.mean(n_pc[1000*count:1000*(count+1)][:,1]),
                              np.mean(n_pc[1000*count:1000*(count+1)][:,2])])
        
        transition_xy = next_mean - current_mean

        transformed_current_pc = transformed_pc + transition_xy

        combined_pc = np.vstack([n_pc, transformed_current_pc])
        return combined_pc

def make_scene():
    def custom_parser():
        # object는 parser.add_argument( ~ , nargs="+") , nargs="+" 때문에 list로 arg 셋팅함
        args = easydict.EasyDict(
            {
                "objects": [
                    # "/home/juju/contact_graspnet/acronym/grasps/Candle_b94fcdffbd1befa57f5e345e9a3e5d44_0.012740999337464653.h5",
                    # "/home/juju/contact_graspnet/acronym/grasps/Canister_714320da4aafcb4a47be2353d2b2403b_0.00023318612778400807.h5",
                    # "/home/juju/contact_graspnet/acronym/grasps/Bowl_95ac294f47fd7d87e0b49f27ced29e3_0.0008357974151618388.h5",
                    # "/home/juju/contact_graspnet/acronym/grasps/Xbox360_435f39e98d2260f0d6e21b8525c3f8bb_0.002061950217848804.h5"
                ],
                "support": "/home/juju/contact_graspnet/acronym/grasps/3Shelves_29b66fc9db2f1558e0e89fd83955713c_0.0025867867973150068.h5",
                "num_grasps": 5,
                "mesh_root": "/home/juju/contact_graspnet/acronym/",
                "support_scale": 0.025,
            }
        )
        return args

    obj_dict = {}

    args = custom_parser()

    args.objects.append("ben_cube.stl")
    # args.objects.append("bottle.stl")
    args.objects.append("can.stl")
    args.objects.append("can.stl")
    args.objects.append("milk.stl")
    args.objects.append("cereal.stl")

    obj_dict = {}
    object_meshes = []
    object_names = []

    for o in args.objects:
        if ".h5" in o:
            object_meshes.append(load_mesh(o, mesh_root_dir=args.mesh_root))
            object_names.append(get_obj_name(obj_dict, o))
        if ".stl" in o:
            if "cube" in o:
                object_meshes.append(get_object_mesh(o, 0.075))
                object_names.append(get_obj_name(obj_dict, o))
            else:
                object_meshes.append(get_object_mesh(o))
                object_names.append(get_obj_name(obj_dict, o))

    # for PYTAMP
    support_mesh = get_object_mesh("ben_table.stl", scale=[1.0, 1.5, 1.0])
    # support_mesh = get_object_mesh("ben_table.stl", scale=[0.8, 1.0, 1.0])
    init_scene = Make_Scene.random_arrangement(
        # object_names, object_meshes, "table", support_mesh
        object_names,
        object_meshes,
        "table",
        support_mesh,
        for_goal_scene=False,
        gaussian=[-0.1, 0, 0.15, 0.15],
    )
    goal_scene = Make_Scene.random_arrangement(
        object_names,
        object_meshes,
        "table",
        support_mesh,
        for_goal_scene=True,
        gaussian=[-0.2, 0, 0.1, 0.15],
    )

    return object_names, init_scene, goal_scene
