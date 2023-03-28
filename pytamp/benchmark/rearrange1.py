import numpy as np
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pytamp.benchmark.benchmark import Benchmark

from pytamp.utils.making_scene_utils import Make_Scene
from pytamp.scene.scene_manager import SceneManager

class Rearrange1(Benchmark):
    def __init__(self, robot_name="panda", 
                 object_names=None, 
                 init_scene:Make_Scene=None, 
                 goal_scene:Make_Scene=None, 
                 geom="visual", 
                 is_pyplot=True):
        """
        
        Args :
            robot_name (str) : robot name to use 
            object_names : object name list on the support object
            scene (Scene for Acronym) : 
        """
        self.object_names = object_names

        self.param = {"object_names": self.object_names, "goal_scene" : goal_scene}
        self.benchmark_config = {0: self.param}
        self.init_scene = init_scene
        self.goal_scene = goal_scene
        self.obj_colors = None
        super().__init__(robot_name, geom, is_pyplot, self.benchmark_config)

        # define goal_scene
        self.goal_scene_mngr = SceneManager(
            self.geom, is_pyplot=self.is_pyplot, benchmark=self.benchmark_config, debug_mode=True
        )

        self._load_robot()
        self._load_objects()
        self._load_scene(self.init_scene, self.scene_mngr)        
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
                [0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi / 4]
            )
        if self.robot_name == "doosan":
            self.robot.setup_link_name("base_0", "right_hand")
            self.robot.init_qpos = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0])

    def _load_objects(self):
        """
            Re-transform the scene created with acronym_tool 
                to fit the pytamp scene with the robot on it 
        """
        self.table_mesh = self.init_scene._support_objects[0]
        
        # Set table_pose according to robot height
        self.table_pose = self.table_pose = Transform(pos=np.array([1.0, -0.6, 0.043]))

        # object_pose transformation according to Table Pose
        for o_name, o_pose in self.init_scene._poses.items():
            self.init_scene._poses[o_name] = self.table_pose.h_mat@o_pose
            
        for o_name, o_pose in self.goal_scene._poses.items():
            self.goal_scene._poses[o_name] = self.table_pose.h_mat@o_pose
        
        # assign color 
        self.init_scene.colorize()

    def _load_scene(self, scene:Make_Scene, scene_mngr:SceneManager):
        """
        Args:
            scene (Make_Scene) : Random scene with object mesh and pose created with Acronym_tool
            scene_mngr (SceneManager) : scene manager for creating specific initial and goal scenes
        """
        scene_mngr.add_object(
            name='table',
            gtype='mesh',
            gparam=scene._objects['support_object'],
            h_mat=scene._poses['support_object'],
            color=[0.823, 0.71, 0.55],
        )

        logical_states = [(f"{o_name}", ("on", "table")) \
                          for o_name in self.object_names]
        
        if not self.obj_colors:
            self.obj_colors = [np.array(scene._objects[o_name].visual.face_colors[:,:3][0])   \
                                for o_name in self.object_names]

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
        scene_mngr.set_logical_state(
            "table", (scene_mngr.scene.logical_state.static, True)
        )
        scene_mngr.set_logical_state(
            scene_mngr.gripper_name, (scene_mngr.scene.logical_state.holding, None)
        )
        scene_mngr.update_logical_states(is_init=False)
        scene_mngr.show_logical_states()

    def render_axis(self, scene_mngr:SceneManager):
        for o_name in self.object_names:
            pose = scene_mngr.scene.objs[o_name].h_mat
            scene_mngr.render.render_axis(pose)
    
