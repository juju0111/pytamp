for i in {1..4}
do
    python3 ../examples/doosan/mcts/benchmark1_test.py --budgets 10 --max_depth 20 --algo bai_perturb --box_number 6 --seed $i &
done
python3 ../examples/doosan/mcts/benchmark1_test.py --budgets 10 --max_depth 20 --algo bai_perturb --box_number 6 --seed 5
