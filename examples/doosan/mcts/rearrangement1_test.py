import numpy as np
import argparse
import os, time

from pytamp.benchmark import Rearrange1
from pykin.utils import plot_utils as p_utils

from pytamp.benchmark.rearrange1 import make_scene
from pytamp.search.mcts_for_rearragement import MCTS_rearrangement

def get_parser():
    parser = argparse.ArgumentParser(description="Test Rearragement 1.")
    parser.add_argument("--budgets", metavar="T", type=int, default=100, help="Horizon")
    parser.add_argument("--max_depth", metavar="H", type=int, default=10, help="Max depth")
    parser.add_argument("--seed", metavar="i", type=int, default=2, help="A random seed")
    parser.add_argument(
        "--algo",
        metavar="alg",
        type=str,
        default="bai_perturb",
        choices=["bai_perturb", "bai_ucb", "uct", "random", "greedy"],
        help="Choose one (bai_perturb, bai_ucb, uct)",
    )
    parser.add_argument(
        "--debug_mode", default=False, type=lambda x: (str(x).lower() == "true"), help="Debug mode"
    )
    parser.add_argument("--box_number", metavar="N", type=int, default=6, help="Box Number(6 or less)")
    args = parser.parse_args()
    return args 

def main():
    args = get_parser() 

    debug_mode = args.debug_mode
    budgets = args.budgets
    max_depth = args.max_depth
    algo = args.algo
    seed = args.seed
    number = args.box_number
    np.random.seed(seed)

    object_names, init_scene, goal_scene = make_scene()
    rearrangement1 = Rearrange1('doosan', object_names, init_scene, goal_scene, is_pyplot=False)

    final_level_1_values = []
    final_level_2_values = []
    final_optimal_nodes = []
    final_pnp_all_joint_paths = []
    final_pick_all_objects = []
    final_place_all_object_poses = []

    # final_optimal_trees = []
    c_list = 10 ** np.linspace(-2, 2.0, 10)

    #######################
    fig, ax = p_utils.init_3d_figure(name="Rearrangement 1")
    # init_scene
    rearrangement1.scene_mngr.render_scene(ax)
    rearrangement1.render_axis(rearrangement1.scene_mngr)
    rearrangement1.scene_mngr.show()


    # goal_scene
    rearrangement1.goal_scene_mngr.render_scene(ax)
    rearrangement1.render_axis(rearrangement1.goal_scene_mngr)
    rearrangement1.goal_scene_mngr.show()

    #######################

    for idx, c in enumerate(c_list):
        
        mcts = MCTS_rearrangement(
                scene_mngr=rearrangement1.scene_mngr,
                init_scene=rearrangement1.init_scene,
                sampling_method=args.algo,
                budgets=args.budgets,
                max_depth=args.max_depth,
                c=c,
                debug_mode=args.debug_mode,
            )

        mcts.only_optimize_1 = True
        start_time = time.time()
        for i in range(budgets):
            print(
                f"\n[{idx+1}/{len(c_list)}] Benchmark: {rearrangement1.scene_mngr.scene.bench_num}, Algo: {algo}, C: {c}, Seed: {seed}"
            )
            mcts.do_planning_rearrange(i)
            
        print("########### Running time : ", time.time()- start_time, "##############")
        final_level_1_values.append(mcts.values_for_level_1)

        ########## level 1 ##########

if __name__=="__main__":
    main()