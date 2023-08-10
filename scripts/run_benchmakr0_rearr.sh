seed=$1
max_depth=$2
use_pick_action=$3
consider_next_scene=$4

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_perturb

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_ucb

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed ${seed} --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo uct

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed 4 --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_perturb

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed 4 --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_ucb

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed 4 --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo uct

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed 5 --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_perturb

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed 5 --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo bai_ucb

python3 -m examples.doosan.mcts.benchmark0_rearr_test --budgets 100 --max_depth ${max_depth} --debug_mode false --obj_num 6 --seed 5 --consider_next_scene ${consider_next_scene} --use_pick_action ${use_pick_action} --algo uct




# 
# bash scripts/run_benchmakr0_rearr.sh {seed} 8 0 1 