
seed=$1
max_depth=$2
use_pick_action=$3
consider_next_scene=$4

n=1

python3 -m examples.doosan.mcts.benchmark1_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --box_number 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_perturb

python3 -m examples.doosan.mcts.benchmark1_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --box_number 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_ucb

python3 -m examples.doosan.mcts.benchmark1_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --box_number 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo uct

python3 -m examples.doosan.mcts.benchmark1_rearr_test --budgets 100 --max_depth 2  --debug_mode false --box_number 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_perturb

python3 -m examples.doosan.mcts.benchmark1_rearr_test --budgets 100 --max_depth 2 --debug_mode false --box_number 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_ucb

python3 -m examples.doosan.mcts.benchmark1_rearr_test --budgets 100 --max_depth 2 --debug_mode false --box_number 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo uct
