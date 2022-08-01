import numpy as np
import argparse
import os

from pykin.utils import plot_utils as p_utils
from pytamp.benchmark import Benchmark2
from pytamp.search.mcts import MCTS


#? python3 benchmark2_test.py --budgets 1000 --max_depth 20 --seed 3 --algo bai_ucb
parser = argparse.ArgumentParser(description='Test Benchmark 2.')
parser.add_argument('--budgets', metavar='T', type=int, default=300, help='Horizon')
parser.add_argument('--max_depth', metavar='H', type=int, default=20, help='Max depth')
parser.add_argument('--seed', metavar='i', type=int, default=1, help='A random seed')
parser.add_argument('--algo', metavar='alg', type=str, default='bai_perturb', choices=['bai_perturb', 'bai_ucb', 'uct'], help='Choose one (bai_perturb, bai_ucb, uct)')
parser.add_argument('--debug_mode', default=False, type=lambda x: (str(x).lower() == 'true'), help='Debug mode')
parser.add_argument('--bottle_number', metavar='N', type=int, default=6, help='Bottle Number(6 or less.)')
args = parser.parse_args()

debug_mode = args.debug_mode
budgets = args.budgets
max_depth = args.max_depth
algo = args.algo
seed = args.seed
number = args.bottle_number
np.random.seed(seed)

benchmark2 = Benchmark2(robot_name="doosan", geom="collision", is_pyplot=True, bottle_num=number)
final_level_1_values = []
final_level_2_values = []
final_pnp_all_joint_paths = []
final_pick_all_objects = []
final_place_all_object_poses = []

c_list = 10**np.linspace(0., 4., 1000)
for idx, c in enumerate(c_list):
    mcts = MCTS(
        scene_mngr=benchmark2.scene_mngr, 
        sampling_method=algo, 
        budgets=budgets, 
        max_depth=max_depth, 
        c=c,
        debug_mode=debug_mode)
    for i in range(budgets):
        print(f"\nBenchmark: {benchmark2.scene_mngr.scene.bench_num}, Algo: {algo}, C: {c}, Seed: {seed}")
        mcts.do_planning(i)

    level_1_max_values = mcts.values_for_level_1
    level_2_max_values = mcts.values_for_level_2
    final_level_1_values.append(mcts.values_for_level_1)
    final_level_2_values.append(mcts.values_for_level_2)

    pnp_all_joint_paths, pick_all_objects, place_all_object_poses = mcts.get_all_joint_path(mcts.optimal_nodes)
    final_pnp_all_joint_paths.append(pnp_all_joint_paths)
    final_pick_all_objects.append(pick_all_objects)
    final_place_all_object_poses.append(place_all_object_poses)

#### File Save ####
pytamp_path = os.path.abspath(os.path.dirname(__file__) + "/../../../")
directory_name = pytamp_path + '/results' + '/benchmark2_result'
p_utils.createDirectory(directory_name)

num = 0
filename = directory_name + '/benchmark2_test_algo({:})_budget({:})_seed({:})_{}.npy'.format(algo, budgets, seed, num)

while os.path.exists(filename):
    filename = directory_name + '/benchmark2_test_algo({:})_budget({:})_seed({:})_{}.npy'.format(algo, budgets, seed, num)
    num += 1

with open(filename, 'wb') as f:
    np.savez(f,
             benchmark_number=benchmark2.scene_mngr.scene.bench_num,
             budgets=budgets,
             max_depth=max_depth,
             algo=algo,
             c=c,
             seed=seed,
             level_1_values=final_level_1_values,
             level_2_values=final_level_2_values,
             pnp_all_joint_paths=final_pnp_all_joint_paths,
             pick_all_objects=final_pick_all_objects,
             place_all_object_poses=final_place_all_object_poses
             )
print('Data saved at {}'.format(filename))