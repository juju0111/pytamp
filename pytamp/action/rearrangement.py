import numpy as np
from collections import OrderedDict
from copy import deepcopy

from pykin.utils import mesh_utils as m_utils
from pykin.utils import plot_utils as p_utils
from pytamp.action.activity import ActivityBase
from pytamp.scene.scene import Scene
from pytamp.utils import heuristic_utils as h_utils
from pykin.utils import transform_utils as t_utils
from pytamp.scene.scene_manager import SceneManager

class RearrangementAction(ActivityBase):
    def __init__(
            self,
            scene_mngr,
    ):
        super().__init__(scene_mngr)

        self.filter_logical_states = [
            scene_mngr.scene.logical_state.support,
            scene_mngr.scene.logical_state.static,
        ]
    
    def get_possible_actions_level_1(self, scene: Scene = None) -> dict:
        """
        Args : 
            scene 
        Return : 
            Possible actions for each object
        """
        # transit object to goal location

        self.deepcopy_scene(scene)

        for obj_name in self.scene_mngr.scene.objs:
            if self.scene_mngr.scene.logical_states[obj_name].get(
                self.scene_mngr.scene.logical_state.on
            ):
                if isinstance(
                    self.scene_mngr.scene.logical_states[obj_name].get(
                        self.scene_mngr.scene.logical_state.on
                    ),
                    list,
                ):
                    for placed_obj in self.scene_mngr.scene.logical_states[obj_name].get(
                        self.scene_mngr.scene.logical_state.on
                    ):
                        placed_obj_name = placed_obj.name

                else:
                    placed_obj_name = (
                        self.scene_mngr.scene.logical_states[obj_name]
                        .get(self.scene_mngr.scene.logical_state.on)
                        .name
                    )
            if not any(
                    logical_state in self.scene_mngr.scene.logical_states[obj_name]
                    for logical_state in self.filter_logical_states
                ):
                action_level_1 = self.get_action_level_1_for_single_object(obj_name=obj_name)
                if not action_level_1:
                    continue
                yield action_level_1

    def get_action_level_1_for_single_object(self, scene=None, obj_name: str = None) -> dict:
        # return possible position
        if scene is not None:
            self.deepcopy_scene(scene)
        
        goal_location = self.get_goal_location(obj_name=obj_name)
        
        # calculate H_matrix from current_pose to goal_pose     
        # c_T_w = t_utils.get_inverse_homogeneous(self.vscene_mngr.scene.objs[obj_name].h_mat)

        grasp_poses_not_collision = list(self.get_goal_location_not_collision(goal_location))
        action = self.get_action(obj_name, grasp_poses_not_collision)
        return action

    def get_goal_location(self, obj_name:str) -> dict:
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")
        
        goal_location = {}
        goal_location[obj_name] = self.scene_mngr.scene.goal_object_poses[obj_name]
        return goal_location

    def get_possible_ik_solve_level_2(self):
        return 0


    def get_possible_joint_path_level_2(self):
        return 0


    def get_possible_transitions(self, scene: Scene = None, action: dict = {}):
        """
        working on table top_scene
        Args:
            scene : scene. 
            action : Assigned object target position
        Returns:
            next_scene : moved 
        """
        print(action)
        if not action:
            ValueError("Not found any action!!")

        for rearr_pose in action[self.info.REARR_POSES]:
            next_scene = deepcopy(scene)

            next_scene.rearr_pose = rearr_pose

            name = action[self.info.REARR_OBJ_NAME]
            pose = rearr_pose[name]

            if pose is not None:
                c_T_w = t_utils.get_inverse_homogeneous(self.scene_mngr.scene.objs[name].h_mat)

                transform_from_cur_to_goal = c_T_w.dot(pose)
                next_scene.transform_from_cut_to_goal = transform_from_cur_to_goal

                # Move object to goal location
                next_scene.objs[name].h_mat = deepcopy(pose)

                # TODO
                self.check_cur_with_goal(scene, name)

            yield next_scene

    # check goal and current transition
    def check_cur_with_goal(self,next_scene:Scene, obj_name:str):
        next_scene.objs[obj_name].h_mat\
            .dot(t_utils.get_inverse_homogeneous(next_scene.goal_object_poses[obj_name]))
        # TODO

    def get_goal_location_not_collision(self, goal_location):
        # self.scene_mngr.set_object_pose(obj_name ,goal_location)
        for obj_name, goal_pose in goal_location.items():
            is_collision = False

            current_location = self.scene_mngr.scene.objs[obj_name].h_mat
            self.scene_mngr.set_object_pose(obj_name,goal_pose)
            result, _ = self.scene_mngr.obj_collision_mngr.in_collision_internal(return_names=True)
            self.scene_mngr.set_object_pose(obj_name,current_location)

            if result:
                is_collision = True
                goal_location[obj_name] = None
            if not is_collision:
                yield goal_location
    
    def get_action(self, obj_name :str, possible_action):
        action = {}
        action[self.info.TYPE] = "rearr"

        action[self.info.REARR_OBJ_NAME] = obj_name
        action[self.info.REARR_POSES] = possible_action
        return action

    def render_axis(self, scene_mngr:SceneManager):
        for o_name in self.object_names:
            pose = scene_mngr.scene.objs[o_name].h_mat
            scene_mngr.render.render_axis(pose)
    