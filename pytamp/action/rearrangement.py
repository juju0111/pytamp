import numpy as np
from collections import OrderedDict
from copy import deepcopy


from pytamp.action.activity import ActivityBase
from pytamp.scene.scene import Scene
from pykin.utils import transform_utils as t_utils
from pytamp.scene.scene_manager import SceneManager

from pytamp.utils.making_scene_utils import Make_Scene


class RearrangementAction(ActivityBase):
    def __init__(self, scene_mngr):
        super().__init__(scene_mngr)

        self.filter_logical_states = [
            scene_mngr.scene.logical_state.support,
            scene_mngr.scene.logical_state.static,
        ]

    def get_possible_actions_level_1(
        self, scene: Scene = None, scene_for_sample: Make_Scene = None
    ) -> dict:
        """
        Args :
            scene :
            scene_for_sample : A class for picking up a random object location
        Return :
            Possible actions for each object

        Use Example)

            rearrangement1 = make_scene_()
            ArrangementAction = RearrangementAction(rearrangement1.scene_mngr)
            actions = list(ArrangementAction.get_possible_actions_level_1(scene_for_sample=rearrangement1.init_scene))

        """
        self.deepcopy_scene(scene)

        # change collision_mngr obj_poses
        if scene:
            for obj_name, obj in scene.objs.items():
                self.scene_mngr.set_object_pose(obj_name, obj.h_mat)

        for obj_name in self.scene_mngr.scene.objs:
            if obj_name == self.scene_mngr.scene.rearr_obj_name:
                continue

            if not any(
                logical_state in self.scene_mngr.scene.logical_states[obj_name]
                for logical_state in self.filter_logical_states
            ):
                action_level_1 = self.get_action_level_1_for_single_object(
                    obj_name=obj_name, scene_for_sample=scene_for_sample
                )
                if not action_level_1:
                    continue
                yield action_level_1

    def get_action_level_1_for_single_object(
        self, obj_name: str = None, scene_for_sample: Make_Scene = None
    ) -> dict:
        """
        A function to get the possible actions for the desired object

        Args 
            obj_name : which object do you want
            scene_for_sample : scene made by Acronym scene.
        Return 
            action : Collision free actions
        """
        # return possible position

        # get_goal_location
        goal_location = self.get_goal_location(obj_name=obj_name)

        # sample_arbitrary_location
        location = list(
            self.get_arbitrary_location(obj_name, scene_for_sample, sample_num=1)
        )
        location.append(goal_location)

        grasp_poses_not_collision = list(self.get_goal_location_not_collision(location))
        action = self.get_action(obj_name, grasp_poses_not_collision)
        return action

    def get_goal_location(self, obj_name: str) -> dict:
        """
        A function that gets a known position from a given goal scene
        
        Args
            obj_name :
        Return
            goal_location :
        """
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")

        goal_location = {}
        goal_location[obj_name] = self.scene_mngr.scene.goal_object_poses[obj_name]
        return goal_location

    def get_arbitrary_location(
        self, obj_name: str, scene_for_sample: Make_Scene = None, sample_num: int = 0
    ) -> dict:
        """
        Sample arbitrary location how many you want

        Args
            obj_name (str)
            scene_for_sample (Make_Scene) : Acronym trimesh scene
            sample_num (int)
        Return
            location : arbitrary location on the table
        """
        total_location = []
        for i in range(sample_num):
            location = {}
            result, pose = scene_for_sample.find_object_placement(
                self.scene_mngr.scene.objs[obj_name].gparam,
                max_iter=100,
                distance_above_support=0.002,
                for_goal_scene=True,
            )
            if result:
                pose = self.scene_mngr.table_pose.h_mat @ pose
                location[obj_name] = pose
                yield location

    def get_possible_ik_solve_level_2(self):
        return 0

    def get_possible_joint_path_level_2(self):
        return 0

    def get_possible_transitions(self, scene: Scene = None, action: dict = {}):
        """
        working on table top_scene 
        Args:
            scene (Scene) : Current scene where the assigned object was not moved
            action (dict) : Assigned object target position
        Returns:
            next_scene : The scene where the assigned object was moved
        """
        if not action:
            ValueError("Not found any action!!")
        name = action[self.info.REARR_OBJ_NAME]
        c_T_w = t_utils.get_inverse_homogeneous(scene.objs[name].h_mat)

        for rearr_pose in action[self.info.REARR_POSES]:
            next_scene = deepcopy(scene)
            next_scene.rearr_obj_name = name
            pose = rearr_pose[name]
            next_scene.rearr_poses = rearr_pose

            next_scene.transform_from_cur_to_goal = c_T_w.dot(pose)

            # Move object to goal location
            next_scene.objs[name].h_mat = deepcopy(pose)

            yield next_scene

    def get_goal_location_not_collision(self, location: list):
        """
        Collision check when an object is moved to a specific location
        Args
            location : want to move to a specific location
        Return
            location : not collide with current scene. 

        """
        for i in location:
            for obj_name, goal_pose in i.items():
                is_collision = False

                current_location = self.scene_mngr.scene.objs[obj_name].h_mat
                self.scene_mngr.set_object_pose(obj_name, goal_pose)
                result, _ = self.scene_mngr.obj_collision_mngr.in_collision_internal(
                    return_names=True
                )
                self.scene_mngr.set_object_pose(obj_name, current_location)

                if result:
                    is_collision = True
                    i[obj_name] = None
                if not is_collision:
                    yield i

    def get_action(self, obj_name: str, possible_action):
        action = {}
        action[self.info.TYPE] = "rearr"

        action[self.info.REARR_OBJ_NAME] = obj_name
        action[self.info.REARR_POSES] = possible_action
        return action

    def render_axis(self, scene_mngr: SceneManager):
        for o_name in self.scene_mngr.scene.goal_objects:
            pose = scene_mngr.scene.objs[o_name].h_mat
            scene_mngr.render.render_axis(pose)
