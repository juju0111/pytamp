import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from copy import deepcopy
from networkx.drawing.nx_agraph import graphviz_layout

import pytamp.utils.sampler as sampler
from pytamp.action.pick import PickAction
from pytamp.action.place import PlaceAction
from pytamp.action.rearrangement import RearrangementAction

from pytamp.search.node_data import NodeData
from pytamp.scene.scene import Scene
from pytamp.scene.scene_manager import SceneManager
from pytamp.utils.making_scene_utils import Make_Scene

from pykin.utils import plot_utils as p_utils
from pykin.utils import mesh_utils as m_utils
from pykin.utils.kin_utils import ShellColors as sc
from pytamp.utils.contact_graspnet_utils import Grasp_Using_Contact_GraspNet


class MCTS_rearrangement:
    def __init__(
        self,
        scene_mngr: SceneManager,
        init_scene: Make_Scene = None,
        sampling_method: str = "uct",
        budgets: int = 500,
        c: float = 100000,
        max_depth: int = 16,
        gamma: float = 1,
        debug_mode=False,
        use_pick_action=False,
        consider_next_scene=True,
        do_level_2=True,
        grasp_use_num: int = 5,
    ):
        self.node_data = NodeData
        self.scene_mngr = scene_mngr
        self.scene_mngr.is_debug_mode = debug_mode
        self.state = scene_mngr.scene
        self.use_pick_action = use_pick_action
        self.consider_next_scene = consider_next_scene
        self._do_level_2 = do_level_2
        self.grasp_use_num = grasp_use_num
        self.prev_rearr_obj_num = 0
        self.next_rearr_obj_num = 0

        bench_num = self.scene_mngr.scene.bench_num

        self.time_used_in_level_1 = 0
        self.time_used_in_level_1_5 = 0
        self.time_used_in_level_2 = 0
        self.max_level_1_value = 0
        self.last_max_level_1_value = 0
        if bench_num == 0:
            # robot action은 우선 패스하고 object transition만 고려해서 Goal Scene을 만족하는
            # state가 나올 때 까지 search!!
            self.rearr_action = RearrangementAction(scene_mngr)
            self.pick_action = PickAction(scene_mngr, n_contacts=5, n_directions=7)
            # self.place_action = PlaceAction(
            #     scene_mngr,
            #     n_samples_held_obj=0,
            #     n_samples_support_obj=0,
            #     n_directions=1,
            #     release_distance=0.02,
            #     retreat_distance=0.15,
            # )
            self.init_scene = init_scene

        if bench_num == 1:
            self.pick_action = PickAction(scene_mngr, n_contacts=0, n_directions=1)
            # self.place_action = PlaceAction(
            #     scene_mngr,
            #     n_samples_held_obj=0,
            #     n_samples_support_obj=0,
            #     n_directions=1,
            #     release_distance=0.02,
            #     retreat_distance=0.15,
            # )
            self.rearr_action = RearrangementAction(scene_mngr)
            self.init_scene = init_scene

        elif bench_num == 2:
            self.pick_action = PickAction(
                scene_mngr,
                n_contacts=0,
                limit_angle_for_force_closure=0.02,
                n_directions=3,
            )
            # self.place_action = PlaceAction(
            #     scene_mngr, n_samples_he
            #     ld_obj=0, n_samples_support_obj=10
            # )
            self.rearr_action = RearrangementAction(scene_mngr, retreat_distance=0.025)
            self.init_scene = init_scene
        elif bench_num == 3:
            self.pick_action = PickAction(
                scene_mngr, n_contacts=0, n_directions=3, retreat_distance=0.15
            )
            # self.place_action = PlaceAction(
            #     scene_mngr,
            #     n_samples_held_obj=0,
            #     n_samples_support_obj=0,
            #     retreat_distance=0.2,
            #     n_directions=3,
            # )
            self.rearr_action = RearrangementAction(scene_mngr)
            self.init_scene = init_scene

        elif bench_num == 4:
            self.pick_action = PickAction(
                scene_mngr, n_contacts=0, n_directions=0, retreat_distance=0.15
            )
            self.place_action = PlaceAction(
                scene_mngr,
                n_samples_held_obj=0,
                n_samples_support_obj=0,
                retreat_distance=0.2,
                n_directions=3,
            )
            self.rearr_action = RearrangementAction(scene_mngr)
            self.init_scene = None

        self._sampling_method = sampling_method
        self._budgets = budgets
        self.c = c
        self.max_depth = max_depth
        self.gamma = gamma
        self.debug_mode = debug_mode

        self.tree = self._create_tree(self.state)
        self.nodes = None

        # benchmark에 적절한 reward setting
        if self.scene_mngr.scene.bench_num == 0:
            # when occupied
            self.infeasible_reward = -10
            # when do appropriate placement
            self.goal_reward = 5

        if self.scene_mngr.scene.bench_num == 1:
            self.infeasible_reward = -10
            self.goal_reward = 5

        if self.scene_mngr.scene.bench_num == 2:
            self.infeasible_reward = -5
            self.goal_reward = 5

        if self.scene_mngr.scene.bench_num == 3:
            self.infeasible_reward = -5
            self.goal_reward = 10
            self.manipulated_obj_num = 0

        # if self.scene_mngr.scene.bench_num == 4:
        #     self.infeasible_reward = -5
        #     self.goal_reward = 15

        self.values_for_level_1 = [0]
        self.values_for_level_2 = [0]
        self.level2_max_value = -np.inf

        self.level_wise_1_success = False
        self.level_wise_2_success = False
        self.infeasible_sub_nodes = []
        self.history_level_1_optimal_nodes = []
        self.history_level_1_values = []
        self.history_level_1_dict = {}
        self.history_level_2_dict = {}

        self.optimal_nodes = []
        self.only_optimize_1 = False
        self.has_aleardy_level_1_optimal_nodes = False
        if not use_pick_action:
            self.grasp_generator = Grasp_Using_Contact_GraspNet(
                self.rearr_action,
                scene_mngr.scene.robot.robot_name,
                self.scene_mngr.scene.bench_num,
            )

    def _create_tree(self, state: Scene):
        tree = nx.DiGraph()
        tree.add_node(0)
        tree.update(
            nodes=[
                (
                    0,
                    {
                        NodeData.DEPTH: 0,
                        NodeData.STATE: state,
                        NodeData.ACTION: None,
                        NodeData.REWARD: 0,
                        NodeData.VALUE: -np.inf,
                        NodeData.VALUE_HISTORY: [],
                        NodeData.VISIT: 0,
                        NodeData.NUMBER: 0,
                        NodeData.TYPE: "state",
                        NodeData.JOINTS: [],
                        NodeData.LEVEL1: False,
                        NodeData.LEVEL2: False,
                        NodeData.LEVEL1_5: False,
                        NodeData.SUCCESS: False,
                        NodeData.COST: 0,
                        NodeData.TEST: (),
                    },
                )
            ]
        )
        return tree

    def do_planning_rearrange(self, iter):
        """
        MCTS level 1 planning for rearrangement

        """
        self.prev_rearr_obj_num = 0
        self.next_rearr_obj_num = 0
        self.pick_obj_set = set()
        self.pick_obj_list = []
        print(f"{sc.HEADER}=========== Search iteration : {iter+1} ==========={sc.ENDC}")
        if self.debug_mode:
            # visited_tree = self.get_visited_subtree()
            # self.visualize_tree("Next Logical Node", self.tree)
            pass
        self.success_level_1_leaf_node = None

        level_1_start_time = time.time()
        self._level_wise_1_optimize_rearr(state_node=0, depth=0)
        self.time_used_in_level_1 += time.time() - level_1_start_time

        ##### modified bugs
        if self.get_max_value_level_1() < self.last_max_level_1_value:
            self.max_level_1_value = self.last_max_level_1_value
        else:
            self.max_level_1_value = self.get_max_value_level_1()

        self.last_max_level_1_value = self.values_for_level_1[-1]
        self.values_for_level_1.append(self.max_level_1_value)

        print("last max  value level 1 :", self.last_max_level_1_value)

        ############  level 1 ################

        if not self.only_optimize_1:
            success_level_1_sub_nodes = None

            # get level 1 node seqeunce
            if self.level_wise_1_success:
                success_level_1_sub_nodes = self.get_nodes_from_leaf_node(
                    self.success_level_1_leaf_node
                )[::-1]

                self.has_aleardy_level_1_optimal_nodes = False
                for level_1_optimal_nodes in self.history_level_1_optimal_nodes:
                    if set(success_level_1_sub_nodes).issubset(level_1_optimal_nodes):
                        print("Aleady has optimal nodes!!")
                        self._revise_values_for_level_1()
                        self.has_aleardy_level_1_optimal_nodes = True
                        break

                ##############  check level 1 already found   ###############

                if not self.has_aleardy_level_1_optimal_nodes:
                    if self.use_pick_action:
                        level_2_start_time = time.time()
                        self._level_wise_2_optimize(success_level_1_sub_nodes)
                        self.time_used_in_level_2 += time.time() - level_2_start_time

                        self._update_success_level_1_and_2(success_level_1_sub_nodes)

                        self.values_for_level_2.append(
                            self.get_max_value_level_2(success_level_1_sub_nodes)
                        )

                    # TODO
                    else:
                        level_1_5_start_time = time.time()
                        self._level_wise_between_1_and_2_optimize(
                            success_level_1_sub_nodes, self.consider_next_scene
                        )
                        self.time_used_in_level_1_5 += time.time() - level_1_5_start_time

                        if self._do_level_2:
                            level_2_start_time = time.time()
                            self._level_wise_2_optimize_rearr(success_level_1_sub_nodes)
                            self.time_used_in_level_2 += time.time() - level_2_start_time

                            self._update_success_level_1_and_2(success_level_1_sub_nodes)

                            print(
                                f"{sc.HEADER} level 2 value : {self.get_max_value_level_2(success_level_1_sub_nodes)} {sc.ENDC}"
                            )
                            self.values_for_level_2.append(
                                self.get_max_value_level_2(success_level_1_sub_nodes)
                            )

                    self.history_level_1_optimal_nodes.append(success_level_1_sub_nodes)
                    self.history_level_1_values.append(
                        self.tree.nodes[0][NodeData.VALUE_HISTORY][-1]
                    )
                    self.history_level_1_dict[len(self.history_level_1_dict)] = dict(
                        nodes=success_level_1_sub_nodes,
                        value=self.tree.nodes[0][NodeData.VALUE_HISTORY][-1],
                    )
                    print("Add level_1_node!")

                    level_2 = True
                    for i, n in enumerate(success_level_1_sub_nodes):
                        if i % 2 == 1:
                            if not self.tree.nodes[success_level_1_sub_nodes[i + 1]].get("joints"):
                                level_2 = False
                    if level_2:
                        self.history_level_2_dict[len(self.history_level_2_dict)] = dict(
                            nodes=success_level_1_sub_nodes,
                            value=self.values_for_level_2[-1],
                        )
                        print("Add level_2_node!")

                else:
                    self.values_for_level_2.append(self.level2_max_value)

                self.level_wise_1_success = False
            else:
                self.values_for_level_2.append(self.level2_max_value)

    def _level_wise_1_optimize_rearr(self, state_node: int, depth: int):
        """
        MCTS level 1 reculsive function

        Args
            state_node (int) : current state node number in tree
            depth (int) : current depth in tree
        return
            value (float) : current state value
        """
        cur_state_node = state_node
        cur_state: Scene = self.tree.nodes[cur_state_node][NodeData.STATE]

        # ? Check Current State
        # *======================================================================================================================== #
        # print("###### check_goal : ",self._is_terminal(cur_state))
        if self._is_terminal(cur_state):
            print(f"{sc.OKBLUE}Success!!!!!{sc.ENDC}")
            self.level_wise_1_success = True
            self.success_level_1_leaf_node = cur_state_node
            reward = self._get_reward(cur_state, depth=depth, is_terminal=True)
            self.tree.nodes[state_node][NodeData.LEVEL1] = True
            self._update_value(cur_state_node, reward)
            return reward

        if depth >= self.max_depth:
            reward = self.infeasible_reward
            self._update_value(cur_state_node, reward)
            # self.render_state("cur_state", cur_state, close_gripper=False)
            print(f"{sc.WARNING}Exceeded the maximum depth!!{sc.ENDC}")
            return 0

        # ? Select Logical Action
        # *======================================================================================================================== #
        cur_logical_action_node = self._select_logical_action_node_rearr(
            cur_state_node, cur_state, depth, self._sampling_method
        )
        cur_logical_action = None

        # #! [DEBUG]
        # if self.debug_mode:
        #     self.visualize_tree("Next Logical Node", self.tree)

        next_state_node = None
        next_state = None

        if cur_logical_action_node is not None:
            cur_logical_action = self.tree.nodes.get(cur_logical_action_node).get(NodeData.ACTION)

            # if cur_logical_action[self.pick_action.info.TYPE] == "pick":
            #     print(
            #         f"{sc.COLOR_BROWN}[Action]{sc.ENDC} {sc.OKGREEN}Pick {cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}"
            #     )
            # if cur_logical_action[self.pick_action.info.TYPE] == "place":
            #     print(
            #         f"{sc.COLOR_BROWN}[Action]{sc.ENDC} {sc.OKGREEN}Place {cur_logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {cur_logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}"
            #     )
            if cur_logical_action[self.rearr_action.info.TYPE] == "rearr":
                print(
                    f"{sc.COLOR_BROWN}[Action]{sc.ENDC} {sc.OKGREEN}Rearr {cur_logical_action[self.rearr_action.info.REARR_OBJ_NAME]} to {cur_logical_action[self.rearr_action.info.PLACE_OBJ_NAME]}{sc.ENDC}"
                )

            # ? Select Next State
            # *======================================================================================================================== #
            next_state_node = self._select_next_state_node(
                cur_logical_action_node,
                cur_state,
                cur_logical_action,
                depth,
                self._sampling_method,
            )
            # assert next_state_node is not None, f"Next state node is None... Why??"
            if next_state_node is not None:
                next_state = self.tree.nodes.get(next_state_node).get(NodeData.STATE)

        #! [DEBUG]
        if self.debug_mode:
            # if cur_logical_action[self.pick_action.info.TYPE] == "pick":
            #     self.render_state("cur_state", cur_state, close_gripper=False)
            #     # self.render_state("next_state", next_state, close_gripper=False)
            # if cur_logical_action[self.pick_action.info.TYPE] == "place":
            #     self.render_state("cur_state", cur_state, close_gripper=True)
            #     # self.render_state("next_state", next_state, close_gripper=True)

            # if cur_logical_action[self.rearr_action.info.TYPE] == "rearr":
            #     self.render_state("cur_state", cur_state, close_gripper=False)
            pass
        # ? Get reward
        # *======================================================================================================================== #
        reward = self._get_reward(cur_state, cur_logical_action, next_state, depth)
        self.prev_rearr_obj_num = self.next_rearr_obj_num
        print(
            f"{sc.MAGENTA}[Reward]{sc.ENDC} S({cur_state_node}) -> A({cur_logical_action_node}) -> S'({next_state_node}) Reward : {sc.UNDERLINE}{np.round(reward,3)}{sc.ENDC}"
        )

        if cur_logical_action_node is None or next_state_node is None:
            value = reward
        else:
            discount_value = -0.1
            if self.scene_mngr.scene.bench_num >= 3:
                discount_value = -0.5
            value = (
                reward
                + discount_value
                + self.gamma * self._level_wise_1_optimize_rearr(next_state_node, depth + 1)
            )

        self._update_value(cur_state_node, value)
        # print(f"{sc.MAGENTA}[Backpropagation]{sc.ENDC} Cur state Node : {cur_state_node}, Value : {np.round(value,3)}")
        # if self.debug_mode:
        #     self.visualize_tree("Backpropagation", self.tree)

        return value

    def _level_wise_between_1_and_2_optimize(self, sub_optimal_nodes, use_next_scene=True) -> None:
        # if self.tree.nodes[0][NodeData.SUCCESS]:
        #     if self.tree.nodes[0][NodeData.VALUE_HISTORY][-1] < self.tree.nodes[0][NodeData.VALUE]:
        #         print(
        #             f"{sc.FAIL}A value of this optimal nodes is lower than maximum value.{sc.ENDC}"
        #         )
        #         self._revise_values_for_level_1()
        #         return

        node_length = int(len(sub_optimal_nodes) / 2)
        for i in range(node_length):
            parent_node = self.tree.nodes[sub_optimal_nodes[2 * i]]
            current_node = self.tree.nodes[sub_optimal_nodes[2 * i + 1]]
            next_node = self.tree.nodes[sub_optimal_nodes[2 * i + 2]]
            obj_to_manipulate = current_node["action"]["rearr_obj_name"]

            if "grasp_poses" in current_node:
                print("already has grasp_poses")
                continue

            print(f"{sc.COLOR_BROWN}{obj_to_manipulate}{sc.ENDC}")
            chance_ = 3
            grasp_poses_not_collision = None
            for _ in range(3):
                if use_next_scene:
                    grasps = self.grasp_generator.get_grasp(
                        init_scene=self.init_scene,
                        next_node=next_node,
                        current_node=current_node,
                    )
                else:
                    grasps = self.grasp_generator.get_grasp(
                        init_scene=self.init_scene,
                        current_node=current_node,
                    )

                if len(grasps) > self.grasp_use_num:
                    random_list = np.random.choice(len(grasps), self.grasp_use_num, replace=False)
                    grasps = grasps[random_list]

                if len(grasps):
                    grasp_poses_not_collision = self.rearr_action.get_all_grasp_poses_not_collision(
                        grasps
                    )

                current_node[self.rearr_action.info.GRASP_SET] = deepcopy(grasps)
                if not grasp_poses_not_collision:
                    current_node[NodeData.LEVEL1_5] = False
                    next_node[NodeData.LEVEL1_5] = False

                if grasp_poses_not_collision is not None:
                    # update next node's grasp_pose

                    current_node[self.rearr_action.info.GRASP_POSES] = grasp_poses_not_collision
                    g_ = random.sample(grasp_poses_not_collision, 1)
                    # print("g_ : ", g_)
                    next_node["action"].update(deepcopy(g_[0]))

                    # update next_node's release_pose
                    transform_bet_gripper_n_obj = m_utils.get_relative_transform(
                        next_node["action"][self.rearr_action.move_data.MOVE_grasp],
                        current_node["state"].objs[obj_to_manipulate].h_mat,
                    )
                    next_node[
                        "state"
                    ].robot.gripper.transform_bet_gripper_n_obj = transform_bet_gripper_n_obj
                    eef_pose = m_utils.get_absolute_transform(
                        transform_bet_gripper_n_obj,
                        next_node["state"].objs[obj_to_manipulate].h_mat,
                    )

                    # release_pose_ik_check = self.check_IK(
                    #     self.scene_mngr.scene.robot.init_qpos, eef_pose
                    # )
                    # if not release_pose_ik_check:
                    #     print(f"{sc.COLOR_RED}failed to compute IK release pose {sc.ENDC}")
                    #     continue

                    release_poses = self.rearr_action.get_all_release_poses(eef_pose)
                    next_node["action"].update(release_poses)

                    parent_node[NodeData.LEVEL1_5] = True
                    current_node[NodeData.LEVEL1_5] = True
                    # next_node[NodeData.LEVEL1_5] = True
                    break
                else:
                    print(
                        f"{sc.COLOR_RED}failed to generate grasp pose for {obj_to_manipulate}{sc.ENDC}"
                    )
                    chance_ -= 1
                    current_node[NodeData.LEVEL1_5] = False
                    next_node[NodeData.LEVEL1_5] = False
            if chance_ == 0:
                # Failed case Reward setting
                # self.tree.nodes[sub_optimal_nodes[2 * i]][NodeData.VALUE] = -current_node.get(
                #     NodeData.VALUE
                # )
                # self.tree.nodes[sub_optimal_nodes[2 * i + 1]][NodeData.VALUE] = -current_node.get(
                #     NodeData.VALUE
                # )
                # self.tree.nodes[sub_optimal_nodes[2 * i + 2]][NodeData.VALUE] = -next_node.get(
                #     NodeData.VALUE
                # )
                self.infeasible_sub_nodes.append(sub_optimal_nodes)
            #     # break

    def _select_logical_action_node_rearr(
        self,
        cur_state_node: int,
        cur_state: Scene,
        depth: int,
        exploration_method: str = "bai_ucb",
    ):
        """
        A function that selects possible action nodes from the current state node

        Args
            cur_state_node : node num in tree
            cur_state : current state's scene
            depth : current state node's depth
            exploration_method
        Return
            logical_action_node : Action Nodes from Current State Nodes

        """
        # e-greedy, softmax
        cur_Visit = self.tree.nodes[cur_state_node][NodeData.VISIT]
        children = [child for child in self.tree.neighbors(cur_state_node)]
        logical_action_node = None

        if not children:
            visit = self.tree.nodes[cur_state_node][NodeData.VISIT]
            if visit == 0:
                # print(f"Current state node {cur_state_node} is a leaf node, So expand")
                self._expand_action_node_rearr(cur_state_node, cur_state, depth)
            expanded_children = [child for child in self.tree.neighbors(cur_state_node)]
            if not expanded_children:
                return logical_action_node
            logical_action_node = np.random.choice(expanded_children)
        else:
            # print(f"Current state node has children {children}")
            logical_action_node = self._sample_child_node(children, exploration_method, depth)

        return logical_action_node

    def _expand_action_node_rearr(self, cur_state_node: int, cur_state: Scene, depth: int):
        """
        A function that creates possible action nodes from the current state node visited for the first time.

        Args
            cur_state_node : node num in tree
            cur_state : current state's scene
            depth : current state node's depth

        """
        if self.use_pick_action:
            ## consider pick & rearr in level_1

            is_holding = (
                cur_state.logical_states[cur_state.robot.gripper.name][
                    cur_state.logical_state.holding
                ]
                is not None
            )

            if not is_holding:
                possible_actions = list(self.pick_action.get_possible_actions_level_1(cur_state))
                # self.render_action("Pick Action", cur_state, possible_actions, is_holding)
            else:
                possible_actions = list(
                    self.rearr_action.get_possible_actions_level_1(
                        scene=cur_state,
                        scene_for_sample=self.init_scene,
                        use_pick_action=self.use_pick_action,
                    )
                )
                # possible_actions = list(self.place_action.get_possible_actions_level_1(cur_state))
                # self.render_action("Place Action", cur_state, possible_actions, is_holding)
        else:
            ## consider only rearr in level_1
            possible_actions = list(
                self.rearr_action.get_possible_actions_level_1(
                    scene=cur_state,
                    scene_for_sample=self.init_scene,
                    use_pick_action=self.use_pick_action,
                )
            )

        for possible_action in possible_actions:
            # print("possible actions :" ,possible_action)
            if possible_action[self.rearr_action.info.TYPE] == "pick":
                action_node = self.tree.number_of_nodes()
                self.tree.add_node(action_node)
                self.tree.update(
                    nodes=[
                        (
                            action_node,
                            {
                                NodeData.DEPTH: depth + 1,
                                NodeData.STATE: cur_state,
                                NodeData.ACTION: possible_action,
                                NodeData.VALUE: -np.inf,
                                NodeData.VALUE_HISTORY: [],
                                NodeData.VISIT: 0,
                                NodeData.NUMBER: action_node,
                                NodeData.TYPE: "action",
                                NodeData.JOINTS: [],
                                NodeData.LEVEL1: False,
                                NodeData.LEVEL2: False,
                                NodeData.LEVEL1_5: False,
                                NodeData.SUCCESS: False,
                                NodeData.COST: 0,
                                NodeData.TEST: (),
                            },
                        )
                    ]
                )
                self.tree.add_edge(cur_state_node, action_node)

            if possible_action[self.rearr_action.info.TYPE] == "rearr":
                for i, action in enumerate(possible_action[self.rearr_action.info.REARR_POSES]):
                    action_ = deepcopy(possible_action)
                    action_[self.rearr_action.info.REARR_POSES] = list()
                    action_[self.rearr_action.info.REARR_POSES].append(action)
                    action_[self.rearr_action.info.PLACE_OBJ_NAME] = possible_action[
                        self.rearr_action.info.PLACE_OBJ_NAME
                    ][i]
                    action_node = self.tree.number_of_nodes()
                    self.tree.add_node(action_node)
                    self.tree.update(
                        nodes=[
                            (
                                action_node,
                                {
                                    NodeData.DEPTH: depth + 1,
                                    NodeData.STATE: cur_state,
                                    NodeData.ACTION: action_,
                                    NodeData.VALUE: -np.inf,
                                    NodeData.VALUE_HISTORY: [],
                                    NodeData.VISIT: 0,
                                    NodeData.NUMBER: action_node,
                                    NodeData.TYPE: "action",
                                    NodeData.JOINTS: [],
                                    NodeData.LEVEL1: False,
                                    NodeData.LEVEL2: False,
                                    NodeData.LEVEL1_5: False,
                                    NodeData.SUCCESS: False,
                                    NodeData.COST: 0,
                                    NodeData.TEST: (),
                                },
                            )
                        ]
                    )
                    self.tree.add_edge(cur_state_node, action_node)

    def _select_next_state_node(
        self,
        cur_logical_action_node: int,
        cur_state: Scene,
        cur_logical_action: dict,
        depth,
        exploration_method="bai_ucb",
    ):
        """
        A function that creates next_state with an action selected from among several actions in the current action node

        Args
            cur_logical_action_node (int) : node num in tree
            cur_state (Scene)
            cur_logical_action (dict) : set of possible actions
            depth (int)
            exploration_method (str)

        """
        next_state_node = None

        children = [child for child in self.tree.neighbors(cur_logical_action_node)]

        if not children:
            visit = self.tree.nodes[cur_logical_action_node][NodeData.VISIT]
            if visit == 0:
                # print(f"Logical action node {cur_logical_action_node} is a leaf node, So expand")
                self._expand_next_state_node_arr(
                    cur_logical_action_node, cur_state, cur_logical_action, depth
                )
            expanded_children = [child for child in self.tree.neighbors(cur_logical_action_node)]
            if not expanded_children:
                self.tree.nodes[cur_logical_action_node][NodeData.VISIT] += 1
                self.tree.nodes[cur_logical_action_node][NodeData.VALUE] = (
                    self.infeasible_reward * 100
                )
                return next_state_node
            next_state_node = np.random.choice(expanded_children)
        else:
            # print(f"Logical action node has children {children}")
            next_state_node = self._sample_child_node(children, exploration_method, depth)

        return next_state_node

    def _expand_next_state_node_arr(
        self,
        cur_logical_action_node: int,
        cur_state: Scene,
        cur_logical_action: dict,
        depth: int,
    ):
        """
        A function that expands the next state node from the first visited action node

        Args
            cur_logical_action_node (int)
            cur_state (Scene)
            cur_logical_action (dict)
            depth (int)

        """
        logical_action_type = cur_logical_action[self.rearr_action.info.TYPE]
        if logical_action_type == "pick":
            next_states = list(
                self.pick_action.get_possible_transitions(cur_state, cur_logical_action)
            )

        if logical_action_type == "place":
            next_states = list(
                self.place_action.get_possible_transitions(cur_state, cur_logical_action)
            )

        if logical_action_type == "rearr":
            next_states = list(
                self.rearr_action.get_possible_transitions(cur_state, cur_logical_action)
            )

        for next_state in next_states:
            next_node = self.tree.number_of_nodes()
            next_scene: Scene = next_state
            if logical_action_type == "pick":
                cur_geometry_action = next_scene.grasp_poses
            if logical_action_type == "place":
                cur_geometry_action = next_scene.release_poses
            if logical_action_type == "rearr":
                cur_geometry_action = next_scene.rearr_poses

            # print("cur_logical_node :", self.tree.nodes[cur_logical_action_node])
            self.tree.add_node(next_node)
            if not self.use_pick_action:
                self.tree.update(
                    nodes=[
                        (
                            next_node,
                            {
                                NodeData.NUMBER: next_node,
                                NodeData.VISIT: 0,
                                NodeData.DEPTH: depth + 1,
                                NodeData.STATE: next_state,
                                NodeData.ACTION: cur_geometry_action,
                                NodeData.VALUE: -np.inf,
                                NodeData.VALUE_HISTORY: [],
                                NodeData.TYPE: "state",
                                NodeData.JOINTS: [],
                                NodeData.LEVEL1: False,
                                NodeData.LEVEL2: False,
                                NodeData.LEVEL1_5: False,
                                NodeData.SUCCESS: False,
                                NodeData.COST: 0,
                                NodeData.TEST: (
                                    cur_logical_action.get("rearr_obj_name"),
                                    cur_logical_action.get("rearr_poses")[0].get(
                                        cur_logical_action.get("place_obj_name")
                                    ),
                                ),
                            },
                        )
                    ]
                )
            else:
                self.tree.update(
                    nodes=[
                        (
                            next_node,
                            {
                                NodeData.NUMBER: next_node,
                                NodeData.VISIT: 0,
                                NodeData.DEPTH: depth + 1,
                                NodeData.STATE: next_state,
                                NodeData.ACTION: cur_geometry_action,
                                NodeData.VALUE: -np.inf,
                                NodeData.VALUE_HISTORY: [],
                                NodeData.TYPE: "state",
                                NodeData.JOINTS: [],
                                NodeData.LEVEL1: False,
                                NodeData.LEVEL2: False,
                                NodeData.LEVEL1_5: False,
                                NodeData.SUCCESS: False,
                                NodeData.COST: 0,
                                NodeData.TEST: (),
                            },
                        )
                    ]
                )
            self.tree.add_edge(cur_logical_action_node, next_node)

    def _sample_child_node(self, children, exploration_method, depth):
        assert len(children) != 0
        if exploration_method == "random":
            best_idx = sampler.find_best_idx_from_random(self.tree, children)
        if exploration_method == "greedy":
            best_idx = sampler.find_idx_from_greedy(self.tree, children)
        if exploration_method == "uct":
            # c = self.c / np.maximum(depth, 1)
            c = self.c
            best_idx = sampler.find_idx_from_uct(self.tree, children, c)
        if exploration_method == "bai_ucb":
            # c = self.c / np.maximum(depth, 1)
            c = self.c
            best_idx = sampler.find_idx_from_bai_ucb(self.tree, children, c)
        if exploration_method == "bai_perturb":
            # c = self.c / np.maximum(depth, 1)
            c = self.c
            best_idx = sampler.find_idx_from_bai_perturb(self.tree, children, c)

        child_node = children[best_idx]

        return child_node

    def _update_value(self, cur_state_node, value):
        self._update_node(cur_state_node, value)
        if cur_state_node != 0:
            action_node = [node for node in self.tree.predecessors(cur_state_node)][0]
            self._update_node(action_node, value)

    def _update_node(self, node, value):
        if node != 0:
            parent_node = [node for node in self.tree.predecessors(node)][0]
            if self.tree.nodes[node][NodeData.LEVEL1] is True:
                self.tree.nodes[parent_node][NodeData.LEVEL1] = True

        self.tree.nodes[node][NodeData.VISIT] += 1
        self.tree.nodes[node][NodeData.VALUE_HISTORY].append(value)
        if value > self.tree.nodes[node][NodeData.VALUE]:
            self.tree.nodes[node][NodeData.VALUE] = value

    @staticmethod
    def _is_terminal(state: Scene):
        """
        Check whether the current state node state is the goal state or not
        """
        if state.is_terminal_state():
            return True
        return False

    def _get_reward(
        self,
        cur_state: Scene = None,
        cur_logical_action: dict = {},
        next_state: Scene = None,
        depth=None,
        is_terminal: bool = False,
    ) -> float:
        reward = -0.1
        if is_terminal:
            print(f"Terminal State! Reward is {self.goal_reward}")
            return self.goal_reward

        # reward function setting
        if self.scene_mngr.scene.bench_num == 0:
            inf_reward = self.infeasible_reward / (max(1, depth)) * 10

        elif self.scene_mngr.scene.bench_num == 1:
            inf_reward = self.infeasible_reward / (max(1, depth)) * 2

        elif self.scene_mngr.scene.bench_num == 2:
            inf_reward = self.infeasible_reward / (max(1, depth)) * 2

        elif self.scene_mngr.scene.bench_num == 3:
            inf_reward = self.infeasible_reward / (max(1, depth)) * 2

        elif self.scene_mngr.scene.bench_num == 4:
            inf_reward = self.infeasible_reward / (max(1, depth)) * 2

        if cur_state is None:
            print(f"Current state is None.. Reward is {inf_reward}")

            return inf_reward

        if cur_logical_action is None:
            print(f"Current logical action is None.. Reward is {inf_reward}")
            return inf_reward

        if next_state is None:
            print(f"Next state is None.. Reward is {inf_reward}")
            return inf_reward

        logical_action_type = cur_logical_action[self.rearr_action.info.TYPE]

        if self.scene_mngr.scene.bench_num == 0:
            # 현재 transition한 object를 받아와서 goal과 비교해야함.
            # self.prev_rearr_obj_num = len(cur_state.rearranged_object)
            if logical_action_type == "rearr":
                next_state_is_success = next_state.check_success_rearr_action(
                    cur_logical_action[self.rearr_action.info.REARR_OBJ_NAME]
                )
                self.next_rearr_obj_num = len(next_state.rearranged_object)

                if self.use_pick_action:
                    if next_state_is_success:
                        # When you place well on your goal
                        if self.next_rearr_obj_num - self.prev_rearr_obj_num == 1:
                            print(f"{sc.COLOR_CYAN}Good Action{sc.ENDC}")
                            return abs(reward) / ((depth + 1)) * 100
                        # When you place object on the target again
                        if self.next_rearr_obj_num - self.prev_rearr_obj_num == 0:
                            print(f"{sc.COLOR_BLUE}not bad Action{sc.ENDC}")
                            return reward
                    else:
                        # When an object in the goal is moved to another place
                        if self.next_rearr_obj_num - self.prev_rearr_obj_num == -1:
                            print(f"{sc.FAIL}Bad Action{sc.ENDC}")
                            return max(reward * 1 / (depth + 1) * 100, self.infeasible_reward)
                        # When an object that was not at the goal position is moved to another location
                        if self.next_rearr_obj_num - self.prev_rearr_obj_num == 0:
                            print(f"{sc.COLOR_BLUE}placed another place not goal{sc.ENDC}")
                            return reward
                else:
                    if next_state_is_success:
                        # When you place well on your goal
                        if self.next_rearr_obj_num - self.prev_rearr_obj_num == 1:
                            print(f"{sc.COLOR_CYAN}Good Action{sc.ENDC}")
                            return abs(reward) / ((depth + 1)) * 50
                        # When you place object on the target again
                        if self.next_rearr_obj_num - self.prev_rearr_obj_num == 0:
                            print(f"{sc.COLOR_BLUE}not bad Action{sc.ENDC}")
                            return reward
                    else:
                        # When an object in the goal is moved to another place
                        if self.next_rearr_obj_num - self.prev_rearr_obj_num == -1:
                            print(f"{sc.FAIL}Bad Action{sc.ENDC}")
                            return max(reward * 1 / (depth + 1) * 50, self.infeasible_reward)
                        # When an object that was not at the goal position is moved to another location
                        if self.next_rearr_obj_num - self.prev_rearr_obj_num == 0:
                            print(f"{sc.COLOR_BLUE}placed another place not goal{sc.ENDC}")
                            return reward
            else:
                # when do pick
                return 0
        elif self.scene_mngr.scene.bench_num == 1:
            prev_stacked_box_num = cur_state.success_stacked_box_num
            next_state_is_success = next_state.check_success_stacked_bench_1()
            # Good action if placed well in alphabetical order

            if self.use_pick_action:
                if logical_action_type == "rearr":
                    if next_state_is_success:
                        if next_state.stacked_box_num - prev_stacked_box_num == 1:
                            print(f"{sc.COLOR_CYAN}Good Action{sc.ENDC}")
                            return abs(reward) * 1 / (depth + 1) * 10
                # repicking object on top of goal_tray is bad
                if logical_action_type == "pick":
                    if next_state.stacked_box_num - prev_stacked_box_num == -1:
                        print(f"{sc.FAIL}Bad Action{sc.ENDC}")
                        return max(reward * 1 / (depth + 1) * 20, self.infeasible_reward)
            else:
                if next_state_is_success:
                    if next_state.stacked_box_num - prev_stacked_box_num == 1:
                        print(f"{sc.COLOR_CYAN}Good Action{sc.ENDC}")
                        return abs(reward) * 1 / (depth + 1) * 100
                if next_state.stacked_box_num - prev_stacked_box_num == -1:
                    print(f"{sc.FAIL}Bad Action{sc.ENDC}")
                    return max(reward * 1 / (depth + 1) * 200, self.infeasible_reward)

        # if self.scene_mngr.scene.bench_num == 2:
        #     logical_action_type = cur_logical_action[self.pick_action.info.TYPE]
        #     if logical_action_type == "place":
        #         pick_obj_y_dis = next_state.get_pose_from_goal_obj(next_state.pick_obj_name)[1, 3]
        #         prev_pick_obj_y_dis = cur_state.get_pose_from_goal_obj(next_state.pick_obj_name)[
        #             1, 3
        #         ]
        #         goal_obj_y_dis = cur_state.get_pose_from_goal_obj("goal_bottle")[1, 3]
        #         dis_between_pick_and_goal_obj = abs(pick_obj_y_dis - goal_obj_y_dis)

        #         if dis_between_pick_and_goal_obj > 0.2:
        #             if pick_obj_y_dis - goal_obj_y_dis < 0:
        #                 if pick_obj_y_dis - prev_pick_obj_y_dis < 0:
        #                     print(f"{sc.COLOR_CYAN}Good Action{sc.ENDC}")
        #                     reward = abs(reward) * 1 / (depth + 1) * 2
        #                 else:
        #                     print(f"{sc.FAIL}Bad Action{sc.ENDC}")
        #                     reward = -1
        #             if pick_obj_y_dis - goal_obj_y_dis > 0:
        #                 if pick_obj_y_dis - prev_pick_obj_y_dis > 0:
        #                     print(f"{sc.COLOR_CYAN}Good Action{sc.ENDC}")
        #                     reward = abs(reward) * 1 / (depth + 1) * 2
        #                 else:
        #                     print(f"{sc.FAIL}Bad Action{sc.ENDC}")
        #                     reward = -1
        #         else:
        #             print(f"{sc.FAIL}Bad Action{sc.ENDC}")
        #             reward = -1
        #         return reward

        #     if logical_action_type == "pick":
        #         cur_pick_obj_name = cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]
        #         if cur_pick_obj_name in self.pick_obj_list:
        #             print(f"{sc.FAIL}Bad Action{sc.ENDC}")
        #             reward = -1
        #         else:
        #             reward = 1
        #         self.pick_obj_list.append(cur_pick_obj_name)

        if self.scene_mngr.scene.bench_num == 3:
            if logical_action_type == "pick":
                cur_pick_obj_name = cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]
                if cur_pick_obj_name in self.pick_obj_set:
                    print(f"{sc.FAIL}Bad Action{sc.ENDC}")
                    reward = -1
                else:
                    reward = 2
                self.pick_obj_set.add(cur_pick_obj_name)

            if logical_action_type == "rearr":
                cur_pick_obj_name = cur_logical_action[self.rearr_action.info.REARR_OBJ_NAME]
                if len(self.pick_obj_set) > 3:
                    print(f"{sc.FAIL}Bad Action{sc.ENDC}")
                    reward = -10
                else:
                    reward = 2
                self.pick_obj_set.add(cur_pick_obj_name)

        # if self.scene_mngr.scene.bench_num == 4:
        #     if logical_action_type == "pick":
        #         cur_pick_obj_name = cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]
        #         if self.pick_obj_list:
        #             if cur_pick_obj_name == self.pick_obj_list[-1]:
        #                 print(f"{sc.FAIL}Bad Action{sc.ENDC}")
        #                 # reward = max(reward * 1/(depth+1) * 20, self.infeasible_reward)
        #                 reward = -1
        #             else:
        #                 print(f"{sc.COLOR_CYAN}Good Action{sc.ENDC}")
        #                 # reward = abs(reward) * 1/(depth+1) * 10
        #                 reward = 2
        #         self.pick_obj_list.append(cur_pick_obj_name)

        # if self.scene_mngr.scene.bench_num == 4:
        #     if logical_action_type == 'place':
        #         cur_obj_pose = next_state.get_pose_from_goal_obj(next_state.pick_obj_name)[:3, 3]
        #         peg_name = next_state.cur_peg_name
        #         peg_pose = next_state.get_pose_from_goal_obj(peg_name)[:3, 3]

        #         print(cur_obj_pose[1], peg_pose[1])
        #         if not np.isclose(cur_obj_pose[1], peg_pose[1], 1e-02):
        #             print(f"{sc.FAIL}Bad Action{sc.ENDC}")
        #             reward = -1
        #         else:
        #             reward = 1

        return reward

    def _revise_values_for_level_1(self):
        self.tree.nodes[0][NodeData.VALUE] = 0
        self.values_for_level_1[-1] = self.last_max_level_1_value
        print("Tree node value : ", self.tree.nodes[0]["value"])
        self.max_level_1_value = 0
        return

    def _level_wise_2_optimize_rearr(self, sub_optimal_nodes):
        if not sub_optimal_nodes:
            print(f"{sc.FAIL}Not found any sub optimal nodes.{sc.ENDC}")
            return

        if self.tree.nodes[0][NodeData.SUCCESS]:
            if self.tree.nodes[0][NodeData.VALUE_HISTORY][-1] < self.tree.nodes[0][NodeData.VALUE]:
                print(
                    f"{sc.FAIL}A value of this optimal nodes is lower than maximum value.{sc.ENDC}"
                )
                self._revise_values_for_level_1()
                return

        for infeasible_node in self.infeasible_sub_nodes:
            if set(sub_optimal_nodes).issubset(infeasible_node):
                print(
                    f"{sc.FAIL}This optimal subnodes({infeasible_node}) is infeasible subnodes.{sc.ENDC}"
                )
                self._revise_values_for_level_1()
                print("Max_value level 1 at failed : ", self.max_level_1_value)
                return

        self.show_logical_actions(sub_optimal_nodes)

        if self.debug_mode:
            # subtree = self.get_success_subtree(optimizer_level=1)
            # self.visualize_tree("Success nodes", subtree)
            pass

        init_thetas = []
        success_pick = False
        success_place = False

        place_scene = None

        for sub_optimal_node in sub_optimal_nodes:
            node_type = (
                "action"
                if self.tree.nodes[sub_optimal_node]["type"] == NodeData.ACTION
                else "state"
            )
            if node_type == "action":
                continue
            if self.tree.nodes[sub_optimal_node].get(NodeData.JOINTS):
                print("Already know path")
                continue

            success_place = False
            success_pick = False

            action: dict = self.tree.nodes[sub_optimal_node].get(NodeData.ACTION)

            if action:
                # next_scene
                pick_scene: Scene = self.tree.nodes[sub_optimal_node]["state"]
                print(f"{sc.COLOR_YELLOW}pick {pick_scene.rearr_obj_name}{sc.ENDC}")
                # print("Grasp_Pose :", action)
                if not init_thetas:
                    init_theta = self.rearr_action.scene_mngr.scene.robot.init_qpos

                grasp_poses = {
                    self.rearr_action.move_data.MOVE_grasp: action[
                        self.rearr_action.move_data.MOVE_grasp
                    ],
                    self.rearr_action.move_data.MOVE_pre_grasp: action[
                        self.rearr_action.move_data.MOVE_pre_grasp
                    ],
                    self.rearr_action.move_data.MOVE_post_grasp: action[
                        self.rearr_action.move_data.MOVE_post_grasp
                    ],
                }

                release_poses = {
                    self.rearr_action.move_data.MOVE_release: action[
                        self.rearr_action.move_data.MOVE_release
                    ],
                    self.rearr_action.move_data.MOVE_pre_release: action[
                        self.rearr_action.move_data.MOVE_pre_release
                    ],
                    self.rearr_action.move_data.MOVE_post_release: action[
                        self.rearr_action.move_data.MOVE_post_release
                    ],
                }
                # pick level 2
                pick_joint_path = self.rearr_action.get_possible_joint_path_level_2_for_grasp(
                    scene=pick_scene,
                    grasp_poses=grasp_poses,
                    init_thetas=init_theta,
                )
                if pick_joint_path:
                    success_pick = True
                    init_theta = pick_joint_path[-1][
                        self.rearr_action.move_data.MOVE_default_grasp
                    ][-1]
                    current_cost = round(self.weird_division(1, self.rearr_action.cost) / 10, 6)
                    if current_cost > self.tree.nodes[sub_optimal_node][NodeData.COST]:
                        self.tree.nodes[sub_optimal_node][NodeData.COST] = current_cost
                        self.tree.nodes[sub_optimal_node][NodeData.JOINTS] = pick_joint_path
                else:
                    print("Pick joint Fail")
                    success_pick = False
                    self.infeasible_sub_nodes.append(sub_optimal_nodes)
                    print(self.infeasible_sub_nodes)
                    break

                # place level 2
                place_joint_path = self.rearr_action.get_possible_joint_path_level_2_for_rearr(
                    scene=pick_scene,
                    release_poses=release_poses,
                    init_thetas=init_theta,
                )
                if place_joint_path:
                    success_place = True
                    init_theta = place_joint_path[-1][
                        self.rearr_action.move_data.MOVE_default_release
                    ][-1]

                    current_cost += round(self.weird_division(1, self.rearr_action.cost) / 10, 6)
                    if current_cost > self.tree.nodes[sub_optimal_node][NodeData.COST]:
                        self.tree.nodes[sub_optimal_node][NodeData.COST] = current_cost
                        self.tree.nodes[sub_optimal_node][NodeData.JOINTS][-1].update(
                            place_joint_path[-1]
                        )

                    parent_node = [node for node in self.tree.predecessors(sub_optimal_node)][0]
                    self.tree.nodes[sub_optimal_node][NodeData.LEVEL2] = True
                    self.tree.nodes[parent_node][NodeData.LEVEL2] = True
                    self.tree.nodes[0][NodeData.LEVEL2] = True

                    # if success_pick and success_place:
                    # print(
                    #     "post processing for video save : Joint path :",
                    #     pick_scene.rearr_obj_name,
                    #     pick_scene.objs[pick_scene.rearr_obj_name].h_mat,
                    # )
                    # self.tree.nodes[sub_optimal_node][NodeData.TEST] = deepcopy(
                    #     (
                    #         pick_scene.rearr_obj_name,
                    #         pick_scene.objs[pick_scene.rearr_obj_name].h_mat,
                    #     )
                    # )
                    print("Success pnp")

                else:
                    print("Place joint Fail")
                    success_pick = False
                    self.infeasible_sub_nodes.append(sub_optimal_nodes)
                    self._revise_values_for_level_1()

                    print(self.infeasible_sub_nodes)
                    break
        if success_pick and success_place:
            self.level_wise_2_success = True

    def _level_wise_2_optimize(self, sub_optimal_nodes):
        if not sub_optimal_nodes:
            print(f"{sc.FAIL}Not found any sub optimal nodes.{sc.ENDC}")
            return

        if self.tree.nodes[0][NodeData.SUCCESS]:
            if self.tree.nodes[0][NodeData.VALUE_HISTORY][-1] < self.tree.nodes[0][NodeData.VALUE]:
                print(
                    f"{sc.FAIL}A value of this optimal nodes is lower than maximum value.{sc.ENDC}"
                )
                self._revise_values_for_level_1()
                return

        for infeasible_node in self.infeasible_sub_nodes:
            if set(sub_optimal_nodes).issubset(infeasible_node):
                print(
                    f"{sc.FAIL}This optimal subnodes({infeasible_node}) is infeasible subnodes.{sc.ENDC}"
                )
                self._revise_values_for_level_1()
                return

        self.show_logical_actions(sub_optimal_nodes)

        # if self.debug_mode:
        #     subtree = self.get_success_subtree(optimizer_level=1)
        #     self.visualize_tree("Success nodes", subtree)

        init_thetas = []
        success_pick = False
        place_scene = None

        for sub_optimal_node in sub_optimal_nodes:
            node_type = (
                "action"
                if self.tree.nodes[sub_optimal_node]["type"] == NodeData.ACTION
                else "state"
            )
            if node_type == "action":
                continue
            success_place = False
            action = self.tree.nodes[sub_optimal_node].get(NodeData.ACTION)
            if action:
                if "grasp" in list(action.keys()):
                    pick_scene: Scene = self.tree.nodes[sub_optimal_node]["state"]
                    print(f"{sc.COLOR_YELLOW}pick {pick_scene.pick_obj_name}{sc.ENDC}")

                    if not init_thetas:
                        init_theta = self.pick_action.scene_mngr.scene.robot.init_qpos

                    pick_joint_path = self.pick_action.get_possible_joint_path_level_2(
                        scene=pick_scene,
                        grasp_poses=pick_scene.grasp_poses,
                        init_thetas=init_theta,
                    )
                    if pick_joint_path:
                        success_pick = True
                        init_theta = pick_joint_path[-1][
                            self.pick_action.move_data.MOVE_default_grasp
                        ][-1]

                        current_cost = round(self.weird_division(1, self.pick_action.cost) / 10, 6)
                        if current_cost > self.tree.nodes[sub_optimal_node][NodeData.COST]:
                            self.tree.nodes[sub_optimal_node][NodeData.COST] = current_cost
                            self.tree.nodes[sub_optimal_node][NodeData.JOINTS] = pick_joint_path

                        parent_node = [node for node in self.tree.predecessors(sub_optimal_node)][0]
                        self.tree.nodes[sub_optimal_node][NodeData.LEVEL2] = True
                        self.tree.nodes[parent_node][NodeData.LEVEL2] = True
                        self.tree.nodes[0][NodeData.LEVEL2] = True
                    else:
                        print("Pick joint Fail")
                        success_pick = False
                        break
                else:
                    place_scene: Scene = self.tree.nodes[sub_optimal_node]["state"]
                    print(
                        f"{sc.COLOR_YELLOW}place {place_scene.pick_obj_name} on {place_scene.cur_place_obj_name}{sc.ENDC}"
                    )

                    place_joint_path = self.rearr_action.get_possible_joint_path_level_2(
                        scene=place_scene,
                        release_poses=place_scene.rearr_poses,
                        init_thetas=init_theta,
                    )
                    if place_joint_path:
                        success_place = True
                        init_theta = place_joint_path[-1][
                            self.rearr_action.move_data.MOVE_default_release
                        ][-1]

                        current_cost = round(self.weird_division(1, self.rearr_action.cost) / 10, 6)
                        if current_cost > self.tree.nodes[sub_optimal_node][NodeData.COST]:
                            self.tree.nodes[sub_optimal_node][NodeData.COST] = current_cost
                            self.tree.nodes[sub_optimal_node][NodeData.JOINTS] = place_joint_path

                        parent_node = [node for node in self.tree.predecessors(sub_optimal_node)][0]
                        self.tree.nodes[sub_optimal_node][NodeData.LEVEL2] = True
                        self.tree.nodes[parent_node][NodeData.LEVEL2] = True
                        self.tree.nodes[0][NodeData.LEVEL2] = True

                        if success_pick and success_place:
                            self.tree.nodes[sub_optimal_node][NodeData.TEST] = deepcopy(
                                (
                                    pick_scene.robot.gripper.attached_obj_name,
                                    place_scene.objs[place_scene.pick_obj_name].h_mat,
                                )
                            )
                            print("Success pnp")
                        else:
                            print("PNP Fail")
                            break
                    else:
                        print("Place joint Fail")
                        success_pick = False
                        break

        if success_pick:
            if self.scene_mngr.scene.bench_num == 2:
                print(f"{sc.COLOR_YELLOW}move {pick_scene.pick_obj_name} to bin{sc.ENDC}")
                if not self.scene_mngr.scene.has_already_final_path:
                    self.scene_mngr.scene.ben_2_final_path = self.place_action.get_rrt_star_path(
                        self.scene_mngr.scene.robot.init_qpos,
                        goal_q=self.scene_mngr.scene.goal_q,
                    )
                    self.scene_mngr.scene.has_already_final_path = True
            self.level_wise_2_success = True
        else:
            self.infeasible_sub_nodes.append(sub_optimal_nodes)
            print(self.infeasible_sub_nodes)

    def _update_success_level_1_and_2(self, sub_optimal_nodes):
        sub_optimal_leaf_node = sub_optimal_nodes[-1]
        success_level_1 = self.tree.nodes[sub_optimal_leaf_node][NodeData.LEVEL1]
        success_level_2 = self.tree.nodes[sub_optimal_leaf_node][NodeData.LEVEL2]

        if success_level_1 and success_level_2:
            for sub_optimal_node in sub_optimal_nodes:
                self.tree.nodes[sub_optimal_node][NodeData.SUCCESS] = True

    def get_nodes_from_leaf_node(self, leaf_node: int):
        parent_nodes = [node for node in self.tree.predecessors(leaf_node)]
        if not parent_nodes:
            return [leaf_node]
        else:
            parent_node = parent_nodes[0]
            return [leaf_node] + self.get_nodes_from_leaf_node(parent_node)

    def get_best_node(self, tree=None, cur_node=0):
        if tree is None:
            tree = self.tree

        if not tree:
            return
        children = [child for child in tree.neighbors(cur_node)]

        if not children:
            return [cur_node]
        else:
            # print(f"Node : {[tree.nodes[child][NodeData.NUMBER] for child in children]}")
            # print(f"Q : {[tree.nodes[child][NodeData.VALUE] for child in children]}")
            # print(f"Visit: {[tree.nodes[child][NodeData.VISIT] for child in children]}")

            best_idx = np.argmax([tree.nodes[child][NodeData.VALUE] for child in children])
            next_node = children[best_idx]
            return [cur_node] + self.get_best_node(tree, next_node)

    def get_minimum_cost_node(self):
        min_cost = 100
        min_cost_nodes = []
        for _, n in self.history_level_2_dict.items():
            cost = 0
            for n_num in n["nodes"]:
                cost += self.tree.nodes[n_num].get("cost")
            if min_cost > cost:
                min_cost = cost
                min_cost_nodes = n["nodes"]
        return min_cost_nodes

    def get_success_subtree(self, optimizer_level=1):
        visited_nodes = []
        if optimizer_level == 1:
            visited_nodes = [
                n for n in self.tree.nodes if self.tree.nodes[n][NodeData.LEVEL1] is True
            ]

        if optimizer_level == 2:
            visited_nodes = [
                n for n in self.tree.nodes if self.tree.nodes[n][NodeData.SUCCESS] is True
            ]

        subtree: nx.DiGraph = self.tree.subgraph(visited_nodes)
        return subtree

    def get_visited_subtree(self):
        visited_nodes = []
        visited_nodes = [n for n in self.tree.nodes if self.tree.nodes[n][NodeData.VISIT] >= 1]

        subtree: nx.DiGraph = self.tree.subgraph(visited_nodes)
        return subtree

    def get_all_leaf_nodes(self, tree: nx.DiGraph):
        leaf_nodes = [node for node in tree.nodes if not [c for c in tree.neighbors(node)]]
        leaf_nodes.sort()
        return leaf_nodes

    def get_max_value_level_1(self):
        max_value = self.tree.nodes[0][NodeData.VALUE]
        return max_value

    def get_max_value_nodes_level_1(self):
        idx = 0
        val_ = -np.inf
        for k, v in self.history_level_1_dict.items():
            if val_ < v["value"]:
                idx = k
                val_ = v["value"]

        max_value_nodes = self.history_level_1_dict[idx]["nodes"]
        return idx, max_value_nodes

    def get_max_value_level_2(self, sub_optimal_nodes):
        value_sum = 0
        if self.tree.nodes[sub_optimal_nodes[-1]][NodeData.SUCCESS]:
            for idx, sub_optimal_node in enumerate(sub_optimal_nodes):
                node_type = (
                    "action"
                    if self.tree.nodes[sub_optimal_node]["type"] == NodeData.ACTION
                    else "state"
                )
                if node_type == NodeData.ACTION:
                    sate_nodes = [child for child in self.tree.neighbors(sub_optimal_node)]
                    if not sate_nodes:
                        break
                    max_state_node_idx = np.argmax(
                        [self.tree.nodes[sate_node][NodeData.VALUE] for sate_node in sate_nodes]
                    )
                    max_state_node = sate_nodes[max_state_node_idx]
                    if self.tree.nodes[max_state_node][NodeData.SUCCESS]:
                        max_state_value = self.tree.nodes[max_state_node][NodeData.VALUE]
                        if (
                            self.tree.nodes[sub_optimal_nodes[idx + 1]][NodeData.VALUE]
                            >= max_state_value
                        ):
                            value_sum += self.tree.nodes[sub_optimal_nodes[idx + 1]][NodeData.COST]
                        else:
                            value_sum = 0
                            break
                if node_type == NodeData.STATE:
                    action_nodes = [child for child in self.tree.neighbors(sub_optimal_node)]
                    if not action_nodes:
                        break
                    max_action_node_idx = np.argmax(
                        [
                            self.tree.nodes[action_node][NodeData.VALUE]
                            for action_node in action_nodes
                        ]
                    )
                    max_action_node = action_nodes[max_action_node_idx]
                    if self.tree.nodes[max_action_node][NodeData.SUCCESS]:
                        max_action_value = self.tree.nodes[max_action_node][NodeData.VALUE]
                        if (
                            self.tree.nodes[sub_optimal_nodes[idx + 1]][NodeData.VALUE]
                            < max_action_value
                        ):
                            value_sum = 0
                            break

        if value_sum == 0:
            return self.level2_max_value
        else:
            print(
                f"{sc.COLOR_GREEN}History : { self.tree.nodes[0][NodeData.VALUE_HISTORY][-1] } value_sum : {value_sum}{sc.ENDC}"
            )
            value_sum = self.tree.nodes[0][NodeData.VALUE_HISTORY][-1] + value_sum
            if self.level2_max_value < value_sum:
                if not set(sub_optimal_nodes).issubset(self.optimal_nodes):
                    self.optimal_nodes = sub_optimal_nodes
                    self.level2_max_value = np.round(value_sum, 6)
                    print(
                        f"{sc.COLOR_CYAN}Update Sub optimal Nodes!! Value is {self.level2_max_value}.{sc.ENDC}"
                    )
            return self.level2_max_value

    def get_grasp_at_current_scene(self, nodes: list, idx: int, consider_next_scene: bool = True):
        current_node = self.tree.nodes[nodes[2 * idx + 1]]
        next_node = self.tree.nodes[nodes[2 * (idx + 1)]]

        if consider_next_scene:
            grasp_poses = self.grasp_generator.get_grasp(
                init_scene=self.init_scene,
                next_node=next_node,
                current_node=current_node,
            )
        else:
            grasp_poses = self.grasp_generator.get_grasp(
                init_scene=self.init_scene,
                next_node=None,
                current_node=current_node,
            )
        return grasp_poses

    def get_all_joint_path(self, nodes):
        pnp_all_joint_path = []
        place_all_object_poses = []
        pick_all_objects = []

        pnp_path = []
        pick_objects = []
        place_object_poses = []

        pick_joint_path = []
        for idx, node in enumerate(nodes):
            node_type = "action" if self.tree.nodes[node]["type"] == NodeData.ACTION else "state"
            if node_type == NodeData.ACTION:
                continue
            action = self.tree.nodes[node].get(NodeData.ACTION)
            if action:
                if self.use_pick_action:
                    if list(action.keys())[0] == "grasp":
                        pick_joint_path = self.tree.nodes[node][NodeData.JOINTS]
                    else:
                        place_joint_path = self.tree.nodes[node][NodeData.JOINTS]
                        pnp_path += pick_joint_path + place_joint_path
                        pick_object, place_obj_pose = self.tree.nodes[node][NodeData.TEST]
                        pick_objects.append(pick_object)
                        place_object_poses.append(place_obj_pose)
                else:
                    pnp_path += self.tree.nodes[node][NodeData.JOINTS]
                    pick_object, place_obj_pose = self.tree.nodes[node][NodeData.TEST]
                    pick_objects.append(pick_object)
                    place_object_poses.append(place_obj_pose)

        if self.scene_mngr.scene.bench_num == 2:
            pnp_path += pick_joint_path
            pnp_path[-1][
                self.pick_action.move_data.MOVE_default_grasp
            ] += self.scene_mngr.scene.ben_2_final_path
            pick_objects.append(self.scene_mngr.scene.goal_object)

        pnp_all_joint_path.append(pnp_path)
        pick_all_objects.append(pick_objects)
        place_all_object_poses.append(place_object_poses)

        return pnp_all_joint_path, pick_all_objects, place_all_object_poses

    def show_logical_actions(self, nodes):
        for node in nodes:
            logical_action = self.tree.nodes[node][NodeData.ACTION]
            if logical_action is not None:
                if self.tree.nodes[node][NodeData.TYPE] == "action":
                    if logical_action[self.pick_action.info.TYPE] == "pick":
                        print(
                            f"Action Node: {node} {sc.OKGREEN}Action: Pick {logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}"
                        )
                    if logical_action[self.pick_action.info.TYPE] == "place":
                        print(
                            f"Action Node: {node}  {sc.OKGREEN}Action: Place {logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}"
                        )

    def visualize_tree(self, title, tree=None):
        if tree is None:
            tree = self.tree
        labels = {}
        for n in tree.nodes:
            if tree.nodes[n][NodeData.ACTION] is not None:
                if tree.nodes[n][NodeData.TYPE] == "action":
                    if self.tree.nodes[n][NodeData.ACTION][self.rearr_action.info.TYPE] == "pick":
                        labels.update(
                            {
                                n: "Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nAction:({} {})\nLevel1:{}\nLevel2:{}".format(
                                    tree.nodes[n][NodeData.TYPE],
                                    tree.nodes[n][NodeData.NUMBER],
                                    tree.nodes[n][NodeData.DEPTH],
                                    tree.nodes[n][NodeData.VISIT],
                                    tree.nodes[n][NodeData.VALUE],
                                    tree.nodes[n][NodeData.ACTION][self.rearr_action.info.TYPE],
                                    tree.nodes[n][NodeData.ACTION][
                                        self.rearr_action.info.PICK_OBJ_NAME
                                    ],
                                    tree.nodes[n][NodeData.LEVEL1],
                                    tree.nodes[n][NodeData.LEVEL2],
                                )
                            }
                        )

                    if tree.nodes[n][NodeData.ACTION][self.rearr_action.info.TYPE] == "place":
                        labels.update(
                            {
                                n: "Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nAction:({} {} on {})\nLevel1:{}\nLevel2:{}".format(
                                    tree.nodes[n][NodeData.TYPE],
                                    tree.nodes[n][NodeData.NUMBER],
                                    tree.nodes[n][NodeData.DEPTH],
                                    tree.nodes[n][NodeData.VISIT],
                                    tree.nodes[n][NodeData.VALUE],
                                    tree.nodes[n][NodeData.ACTION][self.rearr_action.info.TYPE],
                                    tree.nodes[n][NodeData.ACTION][
                                        self.rearr_action.info.HELD_OBJ_NAME
                                    ],
                                    tree.nodes[n][NodeData.ACTION][
                                        self.rearr_action.info.PLACE_OBJ_NAME
                                    ],
                                    tree.nodes[n][NodeData.LEVEL1],
                                    tree.nodes[n][NodeData.LEVEL2],
                                )
                            }
                        )

                    if tree.nodes[n][NodeData.ACTION][self.rearr_action.info.TYPE] == "rearr":
                        labels.update(
                            {
                                n: "Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nAction:({} on {})\nLevel1:{}\nLevel2:{}".format(
                                    tree.nodes[n][NodeData.TYPE],
                                    tree.nodes[n][NodeData.NUMBER],
                                    tree.nodes[n][NodeData.DEPTH],
                                    tree.nodes[n][NodeData.VISIT],
                                    tree.nodes[n][NodeData.VALUE],
                                    tree.nodes[n][NodeData.ACTION][self.rearr_action.info.TYPE],
                                    tree.nodes[n][NodeData.ACTION][
                                        self.rearr_action.info.REARR_OBJ_NAME
                                    ],
                                    tree.nodes[n][NodeData.LEVEL1],
                                    tree.nodes[n][NodeData.LEVEL2],
                                )
                            }
                        )

                if tree.nodes[n][NodeData.TYPE] == "state":
                    labels.update(
                        {
                            n: "Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nLevel1:{}\nLevel2:{}".format(
                                tree.nodes[n][NodeData.TYPE],
                                tree.nodes[n][NodeData.NUMBER],
                                tree.nodes[n][NodeData.DEPTH],
                                tree.nodes[n][NodeData.VISIT],
                                tree.nodes[n][NodeData.VALUE],
                                tree.nodes[n][NodeData.LEVEL1],
                                tree.nodes[n][NodeData.LEVEL2],
                            )
                        }
                    )
            else:
                labels.update(
                    {
                        n: "Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nLevel1:{}\nLevel2:{}".format(
                            tree.nodes[n][NodeData.TYPE],
                            tree.nodes[n][NodeData.NUMBER],
                            tree.nodes[n][NodeData.DEPTH],
                            tree.nodes[n][NodeData.VISIT],
                            tree.nodes[n][NodeData.VALUE],
                            tree.nodes[n][NodeData.LEVEL1],
                            tree.nodes[n][NodeData.LEVEL2],
                        )
                    }
                )

        plt.figure(title, figsize=(14, 10))
        pos = graphviz_layout(tree, prog="dot")
        nx.draw(
            tree,
            pos,
            labels=labels,
            node_shape="s",
            node_color="none",
            bbox=dict(facecolor="skyblue", edgecolor="black", boxstyle="round,pad=0.1"),
        )
        plt.show()

    def render_state(self, title, state: Scene, close_gripper=None):
        ax = None
        if self.scene_mngr.is_pyplot is True:
            fig, ax = p_utils.init_3d_figure(name=title)

        if close_gripper is not None:
            if close_gripper:
                state.robot.close_gripper(0.01)
                if "milk" in state.pick_obj_name:
                    state.robot.close_gripper(0.04)
                if "can" in state.pick_obj_name:
                    state.robot.close_gripper(0.03)
                if "disk" in state.pick_obj_name:
                    state.robot.close_gripper(0.04)
        self.pick_action.scene_mngr.render_objects_and_gripper(ax, state)
        self.pick_action.show()

        if close_gripper is not None:
            if close_gripper:
                state.robot.open_gripper(0.01)
                if "milk" in state.pick_obj_name:
                    state.robot.open_gripper(0.04)
                if "can" in state.pick_obj_name:
                    state.robot.open_gripper(0.03)
                if "disk" in state.pick_obj_name:
                    state.robot.open_gripper(0.04)

    def render_action(self, title, scene, actions, is_holding):
        fig, ax = p_utils.init_3d_figure(name=title)

        if not is_holding:
            for pick_action in actions:
                for grasp_pose in pick_action[self.pick_action.info.GRASP_POSES]:
                    self.pick_action.scene_mngr.render.render_axis(
                        ax, grasp_pose[self.pick_action.move_data.MOVE_grasp]
                    )
                    self.pick_action.scene_mngr.render_gripper(
                        ax, pose=grasp_pose[self.pick_action.move_data.MOVE_grasp]
                    )
        else:
            for place_action in actions:
                for release_pose, obj_pose in place_action[self.place_action.info.RELEASE_POSES]:
                    self.place_action.scene_mngr.render.render_axis(
                        ax, release_pose[self.place_action.move_data.MOVE_release]
                    )
                    self.place_action.scene_mngr.render.render_object(
                        ax,
                        self.place_action.scene_mngr.scene.objs[
                            self.place_action.scene_mngr.scene.robot.gripper.attached_obj_name
                        ],
                        obj_pose,
                        alpha=0.3,
                    )
                    self.place_action.scene_mngr.render_gripper(
                        ax, pose=release_pose[self.place_action.move_data.MOVE_release]
                    )

        self.pick_action.scene_mngr.render_objects(ax, scene)
        p_utils.plot_basis(ax)
        self.pick_action.show()

    def render_rearr_state(self, ax, title, state: Scene):
        if self.scene_mngr.is_pyplot is True:
            fig, ax = p_utils.init_3d_figure(name=title)

        self.rearr_action.scene_mngr.render_objects(ax, state)
        self.rearr_action.render_axis(self.rearr_action.scene_mngr)
        self.rearr_action.show()

    def render_rearr(self, title, nodes: list):
        ax = None
        if self.scene_mngr.is_pyplot is True:
            fig, ax = p_utils.init_3d_figure(name=title)

        self.rearr_action.scene_mngr.scene = self.tree.nodes[nodes[-1]]["state"]

        for i in nodes:
            if self.tree.nodes[i]["type"] == "state":
                self.render_rearr_state(ax, f"{i}", self.tree.nodes[i]["state"])

    def get_visit_node_num(self):
        count = 0
        for i in self.tree.nodes:
            if self.tree.nodes[i]["visit"] > 1:
                count += 1
        return count

    def check_IK(
        self,
        q_pose,
        target_pose,
    ):
        # inverse_kin eef_pose
        success_check_limit = False
        for _ in range(5):
            if success_check_limit:
                break
            q_thetas = self.scene_mngr.scene.robot.inverse_kin(
                q_pose,
                target_pose,
            )

            if not self.rearr_action._check_q_in_limits(q_thetas):
                init_q = np.random.randn(self.scene_mngr.scene.robot.arm_dof)
                continue

            self.scene_mngr.set_robot_eef_pose(q_thetas)
            grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
            pose_error = self.scene_mngr.scene.robot.get_pose_error(target_pose, grasp_pose_from_ik)

            if pose_error < 0.02:
                success_check_limit = True
            else:
                init_q = np.random.randn(self.scene_mngr.scene.robot.arm_dof)
                success_check_limit = False

        return success_check_limit

    @staticmethod
    def weird_division(n, d):
        return round(n / d, 3) if d else 0

    @property
    def sampling_method(self):
        return self._sampling_method

    @sampling_method.setter
    def sampling_method(self, sampling_method):
        self._sampling_method = sampling_method

    @property
    def n_iters(self):
        return self._n_iters

    @n_iters.setter
    def n_iters(self, n_iters):
        self._n_iters = n_iters

    @property
    def budgets(self):
        return self._budgets

    @budgets.setter
    def budgets(self, budgets):
        self._budgets = budgets


if __name__ == "__main__":
    pass
