import numpy as np
import argparse
import matplotlib.pyplot as plt

from pykin.utils import plot_utils as p_utils
from pytamp.benchmark import Benchmark3
from pytamp.search.mcts import MCTS


# #? python3 benchmark2_test.py --budgets 1 --max_depth 1 --seed 3 --algo bai_ucb
parser = argparse.ArgumentParser(description='Test Benchmark 2.')
parser.add_argument('--budgets', metavar='T', type=int, default=300, help='Horizon')
parser.add_argument('--max_depth', metavar='H', type=int, default=20, help='Max depth')
parser.add_argument('--seed', metavar='i', type=int, default=1, help='A random seed')
parser.add_argument('--algo', metavar='alg', type=str, default='bai_perturb', choices=['bai_perturb', 'bai_ucb', 'uct'], help='Sampler Name')
parser.add_argument('--debug_mode', metavar='debug', type=bool, default=False, help='Debug mode')
parser.add_argument('--benchmark', metavar='N', type=int, default=2, help='Benchmark Number')
args = parser.parse_args()

debug_mode = args.debug_mode
budgets = args.budgets
max_depth = args.max_depth
algo = args.algo
seed = args.seed
np.random.seed(seed)

benchmark3 = Benchmark3(robot_name="doosan", geom="collision")
mcts = MCTS(benchmark3.scene_mngr)

mcts.debug_mode = False
mcts.budgets = 300
mcts.max_depth = 20
mcts.c = 300

# mcts.sampling_method = 'bai_ucb' # 405
mcts.sampling_method = 'bai_perturb' # 58
# mcts.sampling_method = 'uct' # 369

for i in range(mcts.budgets):
    mcts.do_planning(i)

subtree = mcts.get_success_subtree(optimizer_level=2)
mcts.visualize_tree("MCTS", subtree)
best_nodes = mcts.get_best_node(subtree)

level_1_max_values = mcts.values_for_level_1
level_2_max_values = mcts.values_for_level_2

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Number of simulations",fontsize=12)
plt.ylabel("Max Value",fontsize=12)
plt.legend(prop={'size' : 12})
plt.plot(level_1_max_values, label="Sum of Values")
plt.plot(level_2_max_values, label="Result Values")
plt.show()

pnp_all_joint_path, pick_all_objects, place_all_object_poses = mcts.get_all_joint_path(mcts.optimal_nodes)
mcts.place_action.simulate_path(pnp_all_joint_path, pick_all_objects, place_all_object_poses)