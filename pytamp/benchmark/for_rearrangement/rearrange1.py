import numpy as np
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils.mesh_utils import get_object_mesh
from pytamp.benchmark.benchmark import Benchmark

class Rearrange1(Benchmark):
    def __init__(self, robot_name="panda", scene=None ,geom="visual", is_pyplot=True):
        """
        
        Args :
            robot_name (str) : robot name to use 
            scene (Scene for Acronym) : 
        """
        # assert box_num <= 6, f"The number of boxes must be 6 or less."
        self.param = {"object_num": self.box_num, "goal_object": "tray_red"}
        self.benchmark_config = {1: self.param}
        self.acronym_scene = scene
        super().__init__(robot_name, geom, is_pyplot, self.benchmark_config)
        self._load_robot()
        self._load_objects()
        self._load_scene()

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
                [0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi / 4]
            )
        if self.robot_name == "doosan":
            self.robot.setup_link_name("base_0", "right_hand")
            self.robot.init_qpos = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0])

    def _load_objects(self):
        