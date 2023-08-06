import numpy as np

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils.mesh_utils import get_object_mesh
from pykin.utils.transform_utils import get_h_mat
from pytamp.benchmark.benchmark import Benchmark

import easydict
from copy import deepcopy

from pytamp.utils.making_scene_utils import load_mesh, get_obj_name, Make_Scene
from pykin.utils.mesh_utils import get_object_mesh
from pykin.utils import mesh_utils as m_utils
from pytamp.scene.scene_manager import SceneManager


class Benchmark2_for_rearr(Benchmark):
    def __init__(self, robot_name="panda", bottle_num=6, geom="visual", is_pyplot=True):
        assert bottle_num <= 8, f"The number of bottles must be 8 or less."
        self.bottle_num = bottle_num
        param = {"bottle_num": self.bottle_num, "goal_object": "goal_bottle"}
        self.benchmark_config = {2: param}
        super().__init__(robot_name, geom, is_pyplot, self.benchmark_config)
        self.make_scene()
        self._load_robot()
        self._load_objects()
        self._load_scene(self.init_scene, self.scene_mngr)

        self.scene_mngr.scene._init_bench_2()

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
                    6.33525628e-13,
                    -5.38478180e-01,
                    5.44820130e-12,
                    -2.69413060e00,
                    2.15158886e-12,
                    2.22111507e00,
                    -7.85398163e-01,
                ]
            )
        if self.robot_name == "doosan":
            self.robot.setup_link_name("base_0", "right_hand")
            self.robot.init_qpos = np.array([0, -np.pi / 6, np.pi / 2 + np.pi / 8, 0, np.pi / 2, 0])

    def _load_objects(self):
        """
        Re-transform the scene created with acronym_tool
            to fit the pytamp scene with the robot on it
        """
        self.shelves_mesh = self.init_scene._support_objects["shelves"]

        self.shelves_pose = Transform(pos=np.array([0.6, 1.4, 1.10]), rot=[0, 0, -np.pi / 2])
        self.bin_pose = Transform(pos=np.array([-0.4, 0.55, 0.0]))
        self.rect_pose = Transform(pos=np.array([0.82, 0.5, 0.615]))

        self.scene_mngr.shelves_pose = self.shelves_pose

        # object_pose transformation according to Table Pose
        for o_name, o_pose in self.init_scene._poses.items():
            # print(o_name ," o_pose : ", o_pose)
            self.init_scene._poses[o_name] = self.shelves_pose.h_mat @ o_pose

    def _load_scene(self, scene: Make_Scene, scene_mngr: SceneManager):
        """
        Args:
            scene (Make_Scene) : Random scene with object mesh and pose created with Acronym_tool
            scene_mngr (SceneManager) : scene manager for creating specific initial and goal scenes
        """
        scene_mngr.add_object(
            name="shelves",
            gtype="mesh",
            gparam=self.shelves_mesh,
            h_mat=scene._poses["shelves"],
            color=[0.823, 0.71, 0.55],
        )

        logical_states = [(f"{o_name}", ("on", "shelves")) for o_name in self.object_names]

        # set object, logical_state in scene_mngr
        for i, o_name in enumerate(self.object_names):
            scene_mngr.add_object(
                name=o_name,
                gtype="mesh",
                gparam=scene._objects[o_name],
                h_mat=scene._poses[o_name],
                color=self.object_color[o_name],
            )
            scene_mngr.set_logical_state(logical_states[i][0], logical_states[i][1])

        # add trash_bin
        self.scene_mngr.add_object(
            name="trash_bin",
            gtype="mesh",
            h_mat=self.bin_pose.h_mat,
            gparam=self.trash_bin_mesh,
            color=[0.64, 0.81, 0.85],
        )

        rect_box_mesh = get_object_mesh("rect_box.stl", [0.013, 0.013, 0.01])
        self.scene_mngr.add_object(
            name="rect_box",
            gtype="mesh",
            h_mat=self.rect_pose.h_mat,
            gparam=rect_box_mesh,
            color=[0.0, 0.0, 0.0],
        )

        scene_mngr.add_robot(self.robot, self.robot.init_qpos)
        scene_mngr.set_logical_state("shelves", (scene_mngr.scene.logical_state.static, True))
        scene_mngr.set_logical_state("trash_bin", (scene_mngr.scene.logical_state.static, True))
        scene_mngr.set_logical_state("rect_box", (scene_mngr.scene.logical_state.static, True))

        scene_mngr.set_logical_state(
            scene_mngr.gripper_name, (scene_mngr.scene.logical_state.holding, None)
        )
        scene_mngr.update_logical_states(is_init=True)
        scene_mngr.show_logical_states()

    def modify_scene(self):
        pass

    def make_scene(
        self,
    ):
        def custom_parser():
            args = easydict.EasyDict(
                {
                    "objects": [
                        "/home/juju/contact_graspnet/acronym/grasps/TrashBin_e3484284e1f301077d9a3c398c7b4709_0.024885138038591482.h5"
                    ],
                    "support": "/home/juju/contact_graspnet/acronym/grasps/3Shelves_22fbb23ca13c345b51887beb710d662a_0.0024196631371593983.h5",
                    "num_grasps": 5,
                    "mesh_root": "/home/juju/contact_graspnet/acronym/",
                    "support_scale": 0.025,
                }
            )
            return args

        obj_dict = {}
        self.object_color = {}
        self.object_names = []
        object_meshes = []
        support_meshes = []
        support_names = []

        args = custom_parser()

        self.trash_bin_mesh = load_mesh(args.objects[0], mesh_root_dir=args.mesh_root, scale=0.055)

        self.bottle_meshes = [
            get_object_mesh("bottle.stl", scale=1.0) for _ in range(self.bottle_num)
        ]
        object_meshes = self.bottle_meshes

        for i in range(len(object_meshes) - 1):
            self.object_names.append(f"bottle_{i}")
            self.object_color[self.object_names[-1]] = [0.0 + i * 0.1, 1.0, 0.0]

        self.object_names.append("goal_bottle")

        self.object_color["goal_bottle"] = [1.0, 0.0, 0.0]

        # for ACRONYM
        support_mesh = load_mesh(
            args.support, mesh_root_dir=args.mesh_root, scale=[0.025, 0.020, 0.0125]
        )

        support_meshes.append(support_mesh)
        support_names.append("shelves")

        self.init_scene = Make_Scene.random_arrangement(
            # self.object_names, object_meshes, "table", support_mesh
            self.object_names,
            object_meshes,
            support_names,
            support_meshes,
            for_goal_scene=True,
            base_mesh="shelves",
            use_distance_limit=False,
        )

        self.init_scene.colorize(specific_objects=self.object_color)
