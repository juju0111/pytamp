import numpy as np
from collections import OrderedDict
from copy import deepcopy
import random
import time
from pykin.utils import mesh_utils as m_utils
from pykin.utils import plot_utils as p_utils
from pytamp.action.activity import ActivityBase
from pytamp.scene.scene import Scene
from pytamp.utils import heuristic_utils as h_utils


class PickAction(ActivityBase):
    """
    n_contacts : The number of samples to grasp the object
    n_directions : The number of heuristically acquiring grasp_direction from the acquired grasp_pose
    """

    def __init__(
        self,
        scene_mngr,
        n_contacts=10,
        n_directions=10,
        limit_angle_for_force_closure=0.2,
        retreat_distance=0.1,
    ):
        super().__init__(scene_mngr)
        self.n_contacts = n_contacts

        if n_directions < 1:
            n_directions = 1
        self.n_directions = n_directions

        self.limit_angle = limit_angle_for_force_closure
        self.retreat_distance = retreat_distance
        self.filter_logical_states = [
            scene_mngr.scene.logical_state.support,
            scene_mngr.scene.logical_state.static,
        ]

    # Expand action to tree
    def get_possible_actions_level_1(self, scene: Scene = None) -> dict:
        self.deepcopy_scene(scene)

        for obj_name in self.scene_mngr.scene.objs:
            # if self.scene_mngr.scene.bench_num != 4:
            if obj_name == self.scene_mngr.scene.pick_obj_name:
                continue

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
                        if self.scene_mngr.scene.bench_num == 2:
                            if placed_obj_name in ["shelf_8", "shelf_15"]:
                                continue
                        if self.scene_mngr.scene.bench_num == 3:
                            if placed_obj_name in ["table"]:
                                continue
                else:
                    placed_obj_name = (
                        self.scene_mngr.scene.logical_states[obj_name]
                        .get(self.scene_mngr.scene.logical_state.on)
                        .name
                    )
                    if self.scene_mngr.scene.bench_num == 2:
                        if placed_obj_name in ["shelf_8", "shelf_15"]:
                            continue
                    if self.scene_mngr.scene.bench_num == 3:
                        if placed_obj_name in ["table"]:
                            continue
            if not any(
                logical_state in self.scene_mngr.scene.logical_states[obj_name]
                for logical_state in self.filter_logical_states
            ):
                action_level_1 = self.get_action_level_1_for_single_object(obj_name=obj_name)
                if not action_level_1[self.info.GRASP_POSES]:
                    continue
                yield action_level_1

        # Expand action to tree

    def get_possible_actions_level_1_even_support(self, scene: Scene = None) -> dict:
        self.deepcopy_scene(scene)

        for obj_name in self.scene_mngr.scene.objs:
            # if self.scene_mngr.scene.bench_num != 4:
            if obj_name == self.scene_mngr.scene.pick_obj_name:
                continue

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
                        if self.scene_mngr.scene.bench_num == 2:
                            if placed_obj_name in ["shelf_8", "shelf_15"]:
                                continue
                        if self.scene_mngr.scene.bench_num == 3:
                            if placed_obj_name in ["table"]:
                                continue
                else:
                    placed_obj_name = (
                        self.scene_mngr.scene.logical_states[obj_name]
                        .get(self.scene_mngr.scene.logical_state.on)
                        .name
                    )
                    if self.scene_mngr.scene.bench_num == 2:
                        if placed_obj_name in ["shelf_8", "shelf_15"]:
                            continue
                    if self.scene_mngr.scene.bench_num == 3:
                        if placed_obj_name in ["table"]:
                            continue
            if self.check_obj_support_static(obj_name):
                action_level_1 = self.get_action_level_1_for_single_object(obj_name=obj_name)
                if not action_level_1[self.info.GRASP_POSES]:
                    continue
                yield action_level_1

    def get_action_level_1_for_single_object(self, scene=None, obj_name: str = None) -> dict:
        if scene is not None:
            self.deepcopy_scene(scene)

        grasp_poses_not_collision = None

        # do this untill find collision free grasp
        for i in range(10):
            if grasp_poses_not_collision:
                break
            grasp_poses = list(self.get_all_grasp_poses(obj_name=obj_name))
            if self.scene_mngr.heuristic:
                grasp_poses.extend(list(self.get_grasp_pose_from_heuristic(obj_name)))
            if not grasp_poses:
                continue
            # print(obj_name, grasp_poses)
            grasp_poses_not_collision = list(self.get_all_grasp_poses_not_collision(grasp_poses))

        action_level_1 = self.get_action(obj_name, grasp_poses_not_collision)
        return action_level_1

    def get_possible_action_level1_for_single_object_even_support(
        self, scene=None, obj_name: str = None
    ) -> dict:
        # In get_possible_action_level_1, filtering is performed if the object to be picked supports another object..!
        # In this function, we can get possible action even support 1 object! but 2 object is not
        self.deepcopy_scene(scene)
        self.check_obj_in_scene(obj_name)

        if self.scene_mngr.scene.pick_obj_name:
            print(
                "picked obj is {}, so you can't pick {}".format(
                    self.scene_mngr.scene.pick_obj_name, obj_name
                )
            )
            return

        # if not any(logical_state in self.scene_mngr.scene.logical_states[obj_name] for logical_state in self.filter_logical_states):
        if self.check_obj_support_static(obj_name):
            action_level_1 = self.get_action_level_1_for_single_object(obj_name=obj_name)
            if not action_level_1[self.info.GRASP_POSES]:
                return
            return action_level_1

    def get_grasp_pose_from_heuristic(self, obj_name):
        copied_mesh = deepcopy(self.scene_mngr.scene.objs[obj_name].gparam)
        copied_mesh.apply_translation(-copied_mesh.center_mass)
        copied_mesh.apply_transform(self.scene_mngr.scene.objs[obj_name].h_mat)
        tcp_poses = h_utils.get_heuristic_tcp_pose(
            scene_mngr=self.scene_mngr,
            object_name=obj_name,
            object_mesh=copied_mesh,
            n_directions=self.n_directions,
        )

        for tcp_pose in tcp_poses:
            grasp_pose = {}
            grasp_pose[
                self.move_data.MOVE_grasp
            ] = self.scene_mngr.scene.robot.gripper.compute_eef_pose_from_tcp_pose(tcp_pose)
            grasp_pose[self.move_data.MOVE_pre_grasp] = self.get_pre_grasp_pose(
                grasp_pose[self.move_data.MOVE_grasp]
            )
            grasp_pose[self.move_data.MOVE_post_grasp] = self.get_post_grasp_pose(
                grasp_pose[self.move_data.MOVE_grasp]
            )

            yield grasp_pose

    # Not Expand, only check possible action using ik
    def get_possible_ik_solve_level_2(self, scene: Scene = None, grasp_poses: dict = {}) -> bool:
        self.deepcopy_scene(scene)

        ik_solve, grasp_poses_filtered = self.compute_ik_solve_for_robot(grasp_poses)
        return ik_solve, grasp_poses_filtered

    def get_possible_joint_path_level_2(
        self, scene: Scene = None, grasp_poses: dict = {}, init_thetas=None
    ):
        self.deepcopy_scene(scene)

        # planning 시작에 앞서 collision manager의 scene을 scene을 transition 이전의 상태로 돌리고 planning 한다.
        pick_obj = self.scene_mngr.scene.robot.gripper.attached_obj_name
        self.scene_mngr.scene.objs[
            pick_obj
        ].h_mat = self.scene_mngr.scene.robot.gripper.pick_obj_pose
        for obj_name in self.scene_mngr.scene.objs:
            obj_pose = self.scene_mngr.scene.objs[obj_name].h_mat
            if obj_name == pick_obj:
                obj_pose = self.scene_mngr.scene.robot.gripper.pick_obj_pose
            self.scene_mngr.obj_collision_mngr.set_transform(obj_name, obj_pose)

        result_all_joint_path = []
        result_joint_path = OrderedDict()
        default_joint_path = []

        default_thetas = init_thetas
        if init_thetas is None:
            default_thetas = self.scene_mngr.scene.robot.init_qpos

        pre_grasp_pose = grasp_poses[self.move_data.MOVE_pre_grasp]
        grasp_pose = grasp_poses[self.move_data.MOVE_grasp]
        post_grasp_pose = grasp_poses[self.move_data.MOVE_post_grasp]
        success_joint_path = True

        # default pose -> pre_grasp_pose (rrt)
        pre_grasp_joint_path = self.get_rrt_star_path(default_thetas, pre_grasp_pose)
        # pre_grasp_joint_path = self.get_prm_star_path(default_thetas, pre_grasp_pose)

        self.cost = 0
        if pre_grasp_joint_path:
            self.cost += self.rrt_planner.goal_node_cost
            # self.cost += self.prm_planner.goal_node_cost

            # pre_grasp_pose -> grasp_pose (cartesian)
            grasp_joint_path = self.get_cartesian_path(pre_grasp_joint_path[-1], grasp_pose)
            if grasp_joint_path:
                # grasp_pose -> post_grasp_pose (cartesian)
                self.scene_mngr.set_robot_eef_pose(grasp_joint_path[-1])
                self.scene_mngr.attach_object_on_gripper(
                    self.scene_mngr.scene.robot.gripper.attached_obj_name
                )

                post_grasp_joint_path = self.get_cartesian_path(
                    grasp_joint_path[-1], post_grasp_pose
                )
                if post_grasp_joint_path:
                    # post_grasp_pose -> default pose (rrt)
                    default_joint_path = self.get_rrt_star_path(
                        post_grasp_joint_path[-1], goal_q=default_thetas
                    )
                else:
                    success_joint_path = False
                    self.scene_mngr.scene[
                        pick_obj
                    ].h_mat = self.scene_mngr.scene.robot.gripper.pick_obj_pose
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
            else:
                success_joint_path = False
        else:
            success_joint_path = False

        if not success_joint_path:
            return result_all_joint_path

        if default_joint_path:
            self.cost += self.rrt_planner.goal_node_cost
            result_joint_path.update({self.move_data.MOVE_pre_grasp: pre_grasp_joint_path})
            result_joint_path.update({self.move_data.MOVE_grasp: grasp_joint_path})
            result_joint_path.update({self.move_data.MOVE_post_grasp: post_grasp_joint_path})
            result_joint_path.update({self.move_data.MOVE_default_grasp: default_joint_path})
            result_all_joint_path.append(result_joint_path)
            return result_all_joint_path

    def get_action(self, obj_name, all_poses):
        action = {}
        action[self.info.TYPE] = "pick"
        action[self.info.PICK_OBJ_NAME] = obj_name
        action[self.info.GRASP_POSES] = all_poses
        return action

    def get_possible_transitions(self, scene: Scene = None, action: dict = {}):
        """
        args    :
            scene   : 변경 전 current scene
            action  : scene을 바꿀 action

        return  :
            next_scene
        """
        if not action:
            ValueError("Not found any action!!")

        pick_obj = action[self.info.PICK_OBJ_NAME]

        for grasp_poses in action[self.info.GRASP_POSES]:
            # 변경 전의 scene
            next_scene = deepcopy(scene)

            ## Change transition
            # scene의 grasp pose(dict)를 대입
            next_scene.grasp_poses = grasp_poses
            # 로봇이 잡아야할 gripper pose를 설정
            next_scene.robot.gripper.grasp_pose = grasp_poses[self.move_data.MOVE_grasp]

            # Gripper Move to grasp pose
            # scene의 gripper_pose를 setting..
            next_scene.robot.gripper.set_gripper_pose(grasp_poses[self.move_data.MOVE_grasp])

            # Get transform between gripper and pick object
            gripper_pose = deepcopy(next_scene.robot.gripper.get_gripper_pose())
            transform_bet_gripper_n_obj = m_utils.get_relative_transform(
                gripper_pose, next_scene.objs[pick_obj].h_mat
            )

            # Attach Object to gripper
            next_scene.pick_obj_name = pick_obj
            next_scene.robot.gripper.attached_obj_name = pick_obj
            next_scene.robot.gripper.pick_obj_pose = deepcopy(next_scene.objs[pick_obj].h_mat)
            next_scene.robot.gripper.transform_bet_gripper_n_obj = deepcopy(
                transform_bet_gripper_n_obj
            )

            # Move a gripper to default pose
            # heuristic setting..!!
            default_thetas = self.scene_mngr.scene.robot.init_qpos
            default_pose = self.scene_mngr.scene.robot.forward_kin(default_thetas)[
                self.scene_mngr.scene.robot.eef_name
            ].h_mat
            next_scene.robot.gripper.set_gripper_pose(default_pose)

            # change pick_obj's h_mat & Move pick object to default pose
            next_scene.objs[pick_obj].h_mat = np.dot(
                next_scene.robot.gripper.get_gripper_pose(),
                next_scene.robot.gripper.transform_bet_gripper_n_obj,
            )
            next_scene.pick_obj_default_pose = deepcopy(next_scene.objs[pick_obj].h_mat)

            ## Change Logical State
            # Remove pick obj in logical state of support obj
            supporting_obj = next_scene.logical_states[pick_obj].get(next_scene.logical_state.on)
            next_scene.prev_place_obj_name = []
            if isinstance(supporting_obj, list):
                for obj in supporting_obj:
                    next_scene.prev_place_obj_name.append(obj.name)
                    next_scene.logical_states.get(obj.name).get(
                        next_scene.logical_state.support
                    ).remove(next_scene.objs[pick_obj])
            else:
                next_scene.prev_place_obj_name.append(supporting_obj.name)
                next_scene.logical_states.get(supporting_obj.name).get(
                    next_scene.logical_state.support
                ).remove(next_scene.objs[pick_obj])

            if self.scene_mngr.scene.bench_num == 4:
                peg_obj = next_scene.logical_states[pick_obj].get(next_scene.logical_state.hang)
                next_scene.prev_peg_name = peg_obj.name
                next_scene.logical_states.get(peg_obj.name).get(
                    next_scene.logical_state.hung
                ).remove(next_scene.objs[pick_obj])

            # Clear logical_state of pick obj
            next_scene.logical_states[pick_obj].clear()

            # Add logical_state of pick obj : {'held' : True}
            next_scene.logical_states[self.scene_mngr.gripper_name][
                next_scene.logical_state.holding
            ] = next_scene.objs[pick_obj]
            next_scene.update_logical_states()
            yield next_scene

    # Not consider collision
    def get_all_grasp_poses(self, obj_name: str) -> dict:
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")

        gripper = self.scene_mngr.scene.robot.gripper
        for tcp_pose in self.get_tcp_poses(obj_name):
            grasp_pose = {}
            grasp_pose[self.move_data.MOVE_grasp] = gripper.compute_eef_pose_from_tcp_pose(tcp_pose)
            grasp_pose[self.move_data.MOVE_pre_grasp] = self.get_pre_grasp_pose(
                grasp_pose[self.move_data.MOVE_grasp]
            )
            grasp_pose[self.move_data.MOVE_post_grasp] = self.get_post_grasp_pose(
                grasp_pose[self.move_data.MOVE_grasp]
            )
            yield grasp_pose

    # for level wise - 1 (Consider gripper collision)
    def get_all_grasp_poses_not_collision(self, grasp_poses):
        if not grasp_poses:
            raise ValueError("Not found grasp poses!")

        for all_grasp_pose in grasp_poses:
            # if self.scene_mngr.scene.bench_num == 2:
            # self.scene_mngr.close_gripper(0.015)
            for name, pose in all_grasp_pose.items():
                is_collision = False

                if name == self.move_data.MOVE_grasp:
                    self.scene_mngr.set_gripper_pose(pose)
                    for name in self.scene_mngr.scene.objs:
                        self.scene_mngr.obj_collision_mngr.set_transform(
                            name, self.scene_mngr.scene.objs[name].h_mat
                        )

                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.move_data.MOVE_pre_grasp:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.move_data.MOVE_post_grasp:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
            # self.scene_mngr.open_gripper(0.015)
            if not is_collision:
                yield all_grasp_pose

    def compute_ik_solve_for_robot(self, grasp_pose: dict):
        ik_solve = {}
        grasp_pose_for_ik = {}

        for name, pose in grasp_pose.items():
            if name == self.move_data.MOVE_grasp:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, grasp_pose_from_ik) and not self._collide(
                    is_only_gripper=False
                ):
                    ik_solve[name] = thetas
                    grasp_pose_for_ik[name] = pose
            if name == self.move_data.MOVE_pre_grasp:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                pre_grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, pre_grasp_pose_from_ik) and not self._collide(
                    is_only_gripper=False
                ):
                    ik_solve[name] = thetas
                    grasp_pose_for_ik[name] = pose
            if name == self.move_data.MOVE_post_grasp:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                post_grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, post_grasp_pose_from_ik) and not self._collide(
                    is_only_gripper=False
                ):
                    ik_solve[name] = thetas
                    grasp_pose_for_ik[name] = pose

        if len(ik_solve) == 3:
            return ik_solve, grasp_pose_for_ik
        return None, None

    def get_contact_points(self, obj_name):
        """
        args :
            obj_name : obj_name
        """
        copied_mesh = deepcopy(self.scene_mngr.scene.objs[obj_name].gparam)
        copied_mesh.apply_transform(self.scene_mngr.scene.objs[obj_name].h_mat)

        center_point = copied_mesh.center_mass

        len_x = abs(center_point[0] - copied_mesh.bounds[0][0])
        len_y = abs(center_point[1] - copied_mesh.bounds[0][1])
        len_z = abs(center_point[2] - copied_mesh.bounds[0][2])

        # weights = self._get_weights_for_held_obj(copied_mesh)

        cnt = 0
        margin = 1
        surface_point_list = []
        start_time = time.time()
        while cnt < self.n_contacts:
            # obj mesh에서 obj_mesh_face에서 weight를 주지 않았으므로 point를 random sample함.
            surface_points, normals = self.get_surface_points_from_mesh(copied_mesh, 2)
            is_success = False

            if self._is_force_closure(surface_points, normals, self.limit_angle):
                if (
                    center_point[0] - len_x * margin
                    <= surface_points[0][0]
                    <= center_point[0] + len_x * margin
                ):
                    if (
                        center_point[1] - len_y * margin
                        <= surface_points[0][1]
                        <= center_point[1] + len_y * margin
                    ):
                        is_success = True

                if (
                    center_point[2] - len_z * margin
                    <= surface_points[0][2]
                    <= center_point[2] + len_z * margin
                ):
                    if (
                        center_point[1] - len_y * margin
                        <= surface_points[0][1]
                        <= center_point[1] + len_y * margin
                    ):
                        is_success = True

                if (
                    center_point[2] - len_z * margin
                    <= surface_points[0][2]
                    <= center_point[2] + len_z * margin
                ):
                    if (
                        center_point[0] - len_x * margin
                        <= surface_points[0][0]
                        <= center_point[0] + len_x * margin
                    ):
                        is_success = True

                if is_success:
                    cnt += 1
                    surface_point_list.append(surface_points)

            if time.time() - start_time > 10:
                return surface_point_list

        return surface_point_list

    @staticmethod
    def _get_weights_for_held_obj(obj_mesh):
        # heuristic
        weights = np.zeros(len(obj_mesh.faces))
        for idx, vertex in enumerate(obj_mesh.vertices[obj_mesh.faces]):
            weights[idx] = 0.05
            if np.all(vertex[:, 2] <= obj_mesh.bounds[0][2] * 1.02):
                weights[idx] = 0.95
        return weights

    def _is_force_closure(self, points, normals, limit_angle):
        # sample된 point가 gripper의 width보다 작고, point가 서로 마주보고있는지!!
        vectorA = points[0]
        vectorB = points[1]

        normalA = -normals[0]
        normalB = -normals[1]

        vectorAB = vectorB - vectorA
        distance = np.linalg.norm(vectorAB)

        unit_vectorAB = m_utils.normalize(vectorAB)
        angle_A2AB = np.arccos(normalA.dot(unit_vectorAB))

        unit_vectorBA = -1 * unit_vectorAB
        angle_B2AB = np.arccos(normalB.dot(unit_vectorBA))

        if distance > self.scene_mngr.scene.robot.gripper.max_width:
            return False

        if angle_A2AB > limit_angle or angle_B2AB > limit_angle:
            return False
        return True

    def get_tcp_poses(self, obj_name):
        contact_points = self.get_contact_points(obj_name)

        for contact_point in contact_points:
            p1, p2 = contact_point
            center_point = (p1 + p2) / 2
            line = p2 - p1

            for _, grasp_dir in enumerate(m_utils.get_grasp_directions(line, self.n_directions)):
                y = m_utils.normalize(line)
                z = grasp_dir

                if abs(np.dot(z, [0, 0, 1])) < 0.5:
                    continue

                # print("grasp_dir : ", grasp_dir)
                x = np.cross(y, z)

                tcp_pose = np.eye(4)
                tcp_pose[:3, 0] = x
                tcp_pose[:3, 1] = y
                tcp_pose[:3, 2] = z
                tcp_pose[:3, 3] = center_point - [0, 0, 0.005]

                yield tcp_pose
