import numpy as np
from collections import OrderedDict
from copy import deepcopy


from pytamp.action.activity import ActivityBase
from pytamp.scene.scene import Scene
from pykin.utils import transform_utils as t_utils
from pykin.utils import mesh_utils as m_utils

from pytamp.scene.scene_manager import SceneManager

from pytamp.utils.making_scene_utils import Make_Scene


class RearrangementAction(ActivityBase):
    def __init__(
        self, scene_mngr, n_sample=1, retreat_distance=0.1, release_distance=0.01
    ):
        super().__init__(scene_mngr)

        self.n_sample = n_sample
        self.filter_logical_states = [
            scene_mngr.scene.logical_state.support,
            scene_mngr.scene.logical_state.static,
        ]
        self.use_pick_action = False

        self.retreat_distance = retreat_distance
        self.release_distance = release_distance

    def get_possible_actions_level_1(
        self,
        scene: Scene = None,
        scene_for_sample: Make_Scene = None,
        use_pick_action=False,
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

        self.use_pick_action = use_pick_action
        # change collision_mngr obj_poses
        if scene:
            for name, obj in scene.objs.items():
                self.scene_mngr.set_object_pose(name, obj.h_mat)

        for obj_name in deepcopy(self.scene_mngr.scene.objs):
            if use_pick_action and (
                obj_name != self.scene_mngr.scene.robot.gripper.attached_obj_name
            ):
                continue

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
        goal_location, goal_sup_obj = self.get_goal_location(obj_name=obj_name)

        # sample_arbitrary_location
        location = list(
            self.get_arbitrary_location(
                obj_name, scene_for_sample, sample_num=self.n_sample
            )
        )
        location.append(goal_location)
        # print(f"first {obj_name} : ", location)

        if self.use_pick_action:
            release_poses_not_collision = list(
                self.get_release_poses_not_collision(obj_name, location)
            )
            action = self.get_action(obj_name, release_poses_not_collision)
        else:
            obj_poses_not_collision = list(
                self.get_goal_location_not_collision(obj_name, location)
            )
            action = self.get_action_only_rearr(obj_name, obj_poses_not_collision)
        
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
        goal_sup_object = "table"
        goal_location[goal_sup_object] = self.scene_mngr.scene.goal_object_poses[
            obj_name
        ]
        return goal_location, goal_sup_object

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
            result, pose, sup_obj_name = scene_for_sample.find_object_placement(
                self.scene_mngr.scene.objs[obj_name].gparam,
                max_iter=100,
                distance_above_support=0.002,
                for_goal_scene=True,
            )
            if result:
                # 현재 rearr action은 Acronym으로 뽑는 방식임
                # 만약 다른 object에 place하고 싶다면 그 object에서 sample point 뽑고
                # point를 sup_obj의 h_mat 위치로 transformation 해주면 됨
                pose = self.scene_mngr.table_pose.h_mat @ pose
                location[sup_obj_name] = pose
                yield location

    def get_possible_ik_solve_level_2(self):
        return 0

    def get_possible_joint_path_level_2(
        self, scene: Scene = None, release_poses: dict = {}, init_thetas=None
    ):
        self.deepcopy_scene(scene)

        result_all_joint_path = []
        result_joint_path = OrderedDict()
        default_joint_path = []

        default_thetas = init_thetas
        if init_thetas is None:
            default_thetas = self.scene_mngr.scene.robot.init_qpos

        pre_release_pose = release_poses[self.move_data.MOVE_pre_release]
        release_pose = release_poses[self.move_data.MOVE_release]
        post_release_pose = release_poses[self.move_data.MOVE_post_release]
        success_joint_path = True

        self.scene_mngr.set_robot_eef_pose(default_thetas)
        self.scene_mngr.set_object_pose(
            scene.pick_obj_name, scene.pick_obj_default_pose
        )
        self.scene_mngr.attach_object_on_gripper(
            self.scene_mngr.scene.robot.gripper.attached_obj_name, True
        )

        pre_release_joint_path = self.get_rrt_star_path(
            default_thetas, pre_release_pose
        )
        self.cost = 0
        if pre_release_joint_path:
            self.cost += self.rrt_planner.goal_node_cost
            # pre_release_pose -> release_pose (cartesian)
            release_joint_path = self.get_cartesian_path(
                pre_release_joint_path[-1], release_pose
            )
            if release_joint_path:
                self.scene_mngr.detach_object_from_gripper()
                self.scene_mngr.add_object(
                    self.scene_mngr.scene.robot.gripper.attached_obj_name,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].gtype,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].gparam,
                    scene.robot.gripper.place_obj_pose,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].color,
                )
                # release_pose -> post_release_pose (cartesian)
                post_release_joint_path = self.get_cartesian_path(
                    release_joint_path[-1], post_release_pose
                )
                if post_release_joint_path:
                    # post_release_pose -> default pose (rrt)
                    default_joint_path = self.get_rrt_star_path(
                        post_release_joint_path[-1], goal_q=default_thetas
                    )
                else:
                    success_joint_path = False
            else:
                success_joint_path = False
        else:
            success_joint_path = False

        if not success_joint_path:
            if self.scene_mngr.is_attached:
                self.scene_mngr.detach_object_from_gripper()
            try:
                self.scene_mngr.add_object(
                    self.scene_mngr.scene.robot.gripper.attached_obj_name,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].gtype,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].gparam,
                    scene.robot.gripper.place_obj_pose,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].color,
                )
            except ValueError as e:
                print(e)
            return result_all_joint_path

        if default_joint_path:
            self.cost += self.rrt_planner.goal_node_cost
            result_joint_path.update(
                {self.move_data.MOVE_pre_release: pre_release_joint_path}
            )
            result_joint_path.update({self.move_data.MOVE_release: release_joint_path})
            result_joint_path.update(
                {self.move_data.MOVE_post_release: post_release_joint_path}
            )
            result_joint_path.update(
                {self.move_data.MOVE_default_release: default_joint_path}
            )
            result_all_joint_path.append(result_joint_path)

            return result_all_joint_path

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

        held_obj_name = action[self.info.REARR_OBJ_NAME]
        place_obj_name = action[self.info.PLACE_OBJ_NAME]

        # print(action)

        if self.use_pick_action:
            for rearr_pose, obj_pose_transformed in action[self.info.REARR_POSES]:
                next_scene = deepcopy(scene)
                next_scene.rearr_poses = rearr_pose

                next_scene.robot.gripper.release_pose = rearr_pose[
                    self.move_data.MOVE_release
                ]

                default_thetas = self.scene_mngr.scene.robot.init_qpos
                default_pose = self.scene_mngr.scene.robot.forward_kin(default_thetas)[
                    self.scene_mngr.scene.robot.eef_name
                ].h_mat
                next_scene.robot.gripper.set_gripper_pose(default_pose)

                next_scene.objs[held_obj_name].h_mat = obj_pose_transformed[
                    place_obj_name
                ]
                self.scene_mngr.obj_collision_mngr.set_transform(
                    held_obj_name, obj_pose_transformed[place_obj_name]
                )

                next_scene.cur_place_obj_name = place_obj_name

                ## Change Logical State
                # Clear logical_state of held obj
                next_scene.logical_states.get(held_obj_name).clear()

                # Chage logical_state holding : None
                next_scene.logical_states[next_scene.robot.gripper.name][
                    next_scene.logical_state.holding
                ] = None

                # Add logical_state of held obj : {'on' : place_obj}
                next_scene.logical_states[held_obj_name][
                    next_scene.logical_state.on
                ] = next_scene.objs[place_obj_name]
    
                yield next_scene

        else:
            for rearr_pose in action[self.info.REARR_POSES]:
                next_scene = deepcopy(scene)
                next_scene.rearr_poses = rearr_pose
                next_scene.rearr_obj_name = name
                pose = rearr_pose[place_obj_name]

                next_scene.transform_from_cur_to_goal = c_T_w.dot(pose)

                # Move object to goal location
                next_scene.objs[name].h_mat = deepcopy(pose)

                yield next_scene

    def get_release_poses_not_collision(self, obj_name: str, location: list):
        """
        Collision check when an object is moved to a specific location
        Args
            location : want to move to a specific location
        Return
            location : not collide with current scene. 

        """
        for i in location:
            for sup_obj_name, goal_pose in i.items():
                is_collision = False

                eef_pose = m_utils.get_absolute_transform(
                    self.scene_mngr.scene.robot.gripper.transform_bet_gripper_n_obj,
                    goal_pose,
                )

                current_location = self.scene_mngr.scene.objs[obj_name].h_mat
                release_poses = self.get_all_release_poses(eef_pose)

                target_obj_and_current_location = {}
                target_obj_and_current_location[sup_obj_name] = goal_pose

                self.scene_mngr.set_object_pose(obj_name, goal_pose)
                result, _ = self.scene_mngr.obj_collision_mngr.in_collision_internal(
                    return_names=True
                )
                # print("middle ",i)
                self.scene_mngr.attach_object_on_gripper(
                    self.scene_mngr.scene.robot.gripper.attached_obj_name, True
                )
                self.scene_mngr.close_gripper()

                for name, pose in release_poses.items():
                    is_collision = False
                    if name == self.move_data.MOVE_release:
                        self.scene_mngr.set_gripper_pose(pose)
                        for name in self.scene_mngr.scene.objs:
                            self.scene_mngr.obj_collision_mngr.set_transform(
                                name, self.scene_mngr.scene.objs[name].h_mat
                            )
                        if self._collide(is_only_gripper=True):
                            is_collision = True
                            break
                    if name == self.move_data.MOVE_pre_release:
                        self.scene_mngr.set_gripper_pose(pose)
                        if self._collide(is_only_gripper=True):
                            is_collision = True
                            break
                    if name == self.move_data.MOVE_post_release:
                        self.scene_mngr.set_gripper_pose(pose)
                        if self._collide(is_only_gripper=True):
                            is_collision = True
                            break

                self.scene_mngr.detach_object_from_gripper()
                self.scene_mngr.add_object(
                    self.scene_mngr.scene.robot.gripper.attached_obj_name,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].gtype,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].gparam,
                    self.scene_mngr.scene.robot.gripper.pick_obj_pose,
                    self.scene_mngr.init_objects[
                        self.scene_mngr.scene.robot.gripper.attached_obj_name
                    ].color,
                )
                self.scene_mngr.open_gripper()

                self.scene_mngr.set_object_pose(obj_name, current_location)

                if result:
                    is_collision = True
                    # i[obj_name] = None

                if not is_collision:
                    yield release_poses, target_obj_and_current_location

    def get_goal_location_not_collision(self, obj_name: str, location: list):
        """
        Collision check when an object is moved to a specific location
        Args
            location : want to move to a specific location
        Return
            location : not collide with current scene. 

        """
        for i in location:
            for sup_obj_name, goal_pose in i.items():
                is_collision = False

                current_location = self.scene_mngr.scene.objs[obj_name].h_mat
                self.scene_mngr.set_object_pose(obj_name, goal_pose)
                result, _ = self.scene_mngr.obj_collision_mngr.in_collision_internal(
                    return_names=True
                )
                self.scene_mngr.set_object_pose(obj_name, current_location)

                if result:
                    is_collision = True
                    # i[obj_name] = None
                if not is_collision:
                    yield i

    def get_all_release_poses(self, eef_pose):
        release_pose = {}
        eef_pose = eef_pose.astype(np.float32)
        release_pose[self.move_data.MOVE_release] = eef_pose
        release_pose[self.move_data.MOVE_pre_release] = self.get_pre_release_pose(
            eef_pose
        )
        release_pose[self.move_data.MOVE_post_release] = self.get_post_release_pose(
            eef_pose
        )
        return release_pose

    def get_pre_release_pose(self, release_pose):
        pre_release_pose = np.eye(4,dtype=np.float32)
        pre_release_pose[:3, :3] = release_pose[:3, :3]
        pre_release_pose[:3, 3] = release_pose[:3, 3] + np.array(
            [0, 0, self.retreat_distance]
        )
        return pre_release_pose

    def get_post_release_pose(self, release_pose):
        post_release_pose = np.eye(4,dtype=np.float32)
        post_release_pose[:3, :3] = release_pose[:3, :3]
        if self.scene_mngr.scene.bench_num != 3:
            post_release_pose[:3, 3] = (
                release_pose[:3, 3] - self.retreat_distance * release_pose[:3, 2]
            )
        else:
            post_release_pose[:3, 3] = release_pose[:3, 3] + np.array(
                [0, 0, self.retreat_distance]
            )
        return post_release_pose

    def get_action(self, obj_name: str, possible_action):
        action = {}
        action[self.info.TYPE] = "rearr"
        action[self.info.REARR_OBJ_NAME] = obj_name

        if possible_action:
            action[self.info.PLACE_OBJ_NAME] = list(possible_action[0][-1].keys())[0]
        else:
            action[self.info.PLACE_OBJ_NAME] = list()

        action[self.info.REARR_POSES] = possible_action
        return action
    
    def get_action_only_rearr(self, obj_name: str, possible_action):
        action = {}
        action[self.info.TYPE] = "rearr"
        action[self.info.REARR_OBJ_NAME] = obj_name
        if possible_action: 
            action[self.info.PLACE_OBJ_NAME] = list(possible_action[-1].keys())[0]
        else:
            action[self.info.PLACE_OBJ_NAME] = list()

        action[self.info.REARR_POSES] = possible_action
        return action

    def render_axis(self, scene_mngr: SceneManager):
        for o_name in self.scene_mngr.scene.goal_objects:
            pose = scene_mngr.scene.objs[o_name].h_mat
            scene_mngr.render.render_axis(pose)
