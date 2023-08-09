for i in 2 3 4 5 6
do
    python3 -m examples.doosan.mcts.benchmark1_rearr_test --budgets 100 --max_depth 16 --algo random --debug_mode false --box_number 6 --seed $i --consider_next_scene 1 --use_pick_action 0  &

    sleep 3
done
# python3 -m examples.doosan.mcts.benchmark1_rearr_test --budgets 100 --max_depth 16 --algo random --debug_mode false --box_number 6 --seed 10 --consider_next_scene 1 --use_pick_action 0
