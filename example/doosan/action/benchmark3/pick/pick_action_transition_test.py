from pykin.utils import plot_utils as p_utils
from pytamp.action.pick import PickAction
from pytamp.benchmark import Benchmark2

benchmark2 = Benchmark2(robot_name="doosan", geom="visual", is_pyplot=True, bottle_num=6)
pick = PickAction(benchmark2.scene_mngr, n_contacts=0, n_directions=0, retreat_distance=0.1)

################# Action Test ##################
actions = list(pick.get_possible_actions_level_1())
for action in actions:
    for idx, pick_scene in enumerate(pick.get_possible_transitions(pick.scene_mngr.scene, action=action)):
        fig, ax = p_utils.init_3d_figure( name="all possible transitions")
        pick.scene_mngr.render_gripper(ax, pick_scene, alpha=0.9, only_visible_axis=False)
        pick.scene_mngr.render_objects(ax, pick_scene)
        pick_scene.show_logical_states()
        ax.view_init(20, 150, 'z')
        pick.scene_mngr.show()