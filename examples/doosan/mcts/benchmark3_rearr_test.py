import numpy as np
import argparse
import os, time

from pykin.utils import plot_utils as p_utils

from pytamp.benchmark import Benchmark3_for_rearr as Benchmark3
from pytamp.search.mcts_for_rearragement import MCTS_rearrangement 


# ? python3 benchmark2_test.py --budgets 100 --max_depth 14 --seed 1 --algo bai_perturb
parser = argparse.ArgumentParser(description="Test Benchmark 3.")
parser.add_argument("--budgets", metavar="T", type=int, default=100, help="Horizon")
parser.add_argument("--max_depth", metavar="H", type=int, default=12, help="Max depth")
parser.add_argument("--seed", metavar="i", type=int, default=16, help="A random seed")
parser.add_argument(
    "--algo",
    metavar="alg",
    type=str,
    default="uct",
    choices=["bai_perturb", "bai_ucb", "uct", "random", "greedy"],
    help="Choose one (bai_perturb, bai_ucb, uct)",
)
parser.add_argument(
    "--grasp_generator",
    metavar="grasp_generator",
    type=str,
    default="None",
    choices=["contact_graspnet", "scale_balance_grasp", "fgc_grasp"],
    help="Choose one (contact_graspnet, scale_balance_grasp, fgc_grasp)",
)
parser.add_argument(
    "--debug_mode",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
    help="Debug mode",
)
parser.add_argument(
    "--grasp_use_num", metavar="N", type=int, default=7, help="How many grasp consider ?"
)
parser.add_argument(
    "--consider_next_scene",
    metavar="N",
    type=int,
    default=1,
    help="Consider Next_scene for generate grasps",
)
parser.add_argument(
    "--use_pick_action",
    metavar="N",
    type=int,
    default=0,
    help="use pick action? (True) or use contact_graspnet (False)",
)

args = parser.parse_args()

debug_mode = args.debug_mode
budgets = args.budgets
max_depth = args.max_depth
algo = args.algo
seed = args.seed
np.random.seed(seed)


final_level_1_values = []
final_level_2_values = []
final_optimal_nodes = []
final_pnp_all_joint_paths = []
final_pick_all_objects = []
final_place_all_object_poses = []

### new
final_used_time = []
final_visited_node_num = []
final_visited_node_num_each_node = []

c_list = 10 ** np.linspace(-2, 2.0, 10)
pass_idx = [0,1,2,8,9]


flag = 0

if (not args.use_pick_action) and args.consider_next_scene:
    flag = 2
elif (not args.use_pick_action) and (not args.consider_next_scene):
    flag = 1
elif args.use_pick_action:
    flag = 0

for idx, c in enumerate(c_list):
    if idx in pass_idx:
        print("Pass Idx")
        final_pnp_all_joint_paths.append([])
        final_pick_all_objects.append([])
        final_place_all_object_poses.append([])
        final_optimal_nodes.append([])
        # final_optimal_trees.append(mcts.tree.nodes)

        ##
        final_used_time.append(
            []
        )
        final_visited_node_num.append([])
        final_visited_node_num_each_node.append([])
    else:
        benchmark3 = Benchmark3(robot_name="panda", geom="collision", is_pyplot=False)

        mcts = MCTS_rearrangement(
            scene_mngr=benchmark3.scene_mngr,
            init_scene = benchmark3.init_scene,
            sampling_method=algo,
            budgets=budgets,
            max_depth=max_depth,
            grasp_generator_name=args.grasp_generator,
            c=c,
            debug_mode=args.debug_mode,
            use_pick_action=args.use_pick_action,
            consider_next_scene=args.consider_next_scene,
            grasp_use_num=args.grasp_use_num,
        
        )

        mcts.only_optimize_1 = False

        start_time = time.time()
        for i in range(budgets):
            print(
                f"\n[{idx+1}/{len(c_list)}] Benchmark: {benchmark3.scene_mngr.scene.bench_num}, Algo: {algo}, C: {c}, Seed: {seed}"
            )
            mcts.do_planning_rearrange(i)

        print("########### Running time : ", time.time() - start_time, "##############")
        final_level_1_values.append(mcts.values_for_level_1)
        final_level_2_values.append(mcts.values_for_level_2)

        if mcts.level_wise_2_success:
            (
                pnp_all_joint_paths,
                pick_all_objects,
                place_all_object_poses,
            ) = mcts.get_all_joint_path(mcts.optimal_nodes)
            final_pnp_all_joint_paths.append(pnp_all_joint_paths)
            final_pick_all_objects.append(pick_all_objects)
            final_place_all_object_poses.append(place_all_object_poses)
            final_optimal_nodes.append(mcts.optimal_nodes)

            ##
            final_used_time.append(
                [mcts.time_used_in_level_1, mcts.time_used_in_level_1_5, mcts.time_used_in_level_2]
            )
            final_visited_node_num.append([mcts.get_visit_node_num()])
            final_visited_node_num_each_node.append(mcts.get_visit_node_num_each_depth())

        else:
            final_pnp_all_joint_paths.append([])
            final_pick_all_objects.append([])
            final_place_all_object_poses.append([])
            final_optimal_nodes.append([])
            # final_optimal_trees.append(mcts.tree.nodes)

            ##
            final_used_time.append(
                [mcts.time_used_in_level_1, mcts.time_used_in_level_1_5, mcts.time_used_in_level_2]
            )
            final_visited_node_num.append([mcts.get_visit_node_num()])
            final_visited_node_num_each_node.append(mcts.get_visit_node_num_each_depth())

            # del mcts
            # print(final_optimal_trees)
            print("delete mcts")

#### File Save ####
pytamp_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + "/../../../")
directory_name = pytamp_path + "/results" + "/benchmark3" + "/benchmark3_rearr_result_new"
p_utils.createDirectory(directory_name)

num = 0
filename = (
    directory_name
    + "/benchmark3_rearr_test_algo({:})_budget({:})_seed({:})_obj({})_flag({})_{}_{}.npy".format(
        algo, budgets, seed, 0, flag, args.grasp_generator,num
    )
)

while os.path.exists(filename):
    filename = (
        directory_name
        + "/benchmark3_rearr_test_algo({:})_budget({:})_seed({:})_obj({})_flag({})_{}_{}.npy".format(
            algo, budgets, seed, 0, flag,args.grasp_generator, num
        )
    )
    num += 1

with open(filename, "wb") as f:
    np.savez(
        f,
        benchmark_number=benchmark3.scene_mngr.scene.bench_num,
        budgets=budgets,
        max_depth=max_depth,
        algo=algo,
        c=c_list,
        seed=seed,
        level_1_values=final_level_1_values,
        level_2_values=final_level_2_values,
        pnp_all_joint_paths=final_pnp_all_joint_paths,
        pick_all_objects=final_pick_all_objects,
        place_all_object_poses=final_place_all_object_poses,
        optimal_nodes=final_optimal_nodes,
        #  optimal_trees=final_optimal_trees
        final_used_time=final_used_time,
        final_visited_node_num=final_visited_node_num,
        final_visited_node_num_each_node = final_visited_node_num_each_node,
        used_obj_num=[0],
    )
print("Data saved at {}".format(filename))
