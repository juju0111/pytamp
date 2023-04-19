from pytamp.benchmark import Rearrange1
from pykin.utils import plot_utils as p_utils

from pytamp.action.rearrangement import RearrangementAction
from pytamp.benchmark.rearrange1 import make_scene


def main():
    object_names, init_scene, goal_scene = make_scene()

    rearrangement1 = Rearrange1(
        "doosan", object_names, init_scene, goal_scene, is_pyplot=False
    )
    ArrangementAction = RearrangementAction(rearrangement1.scene_mngr)

    ################# Action Test ##################
    rearr_actions = list(
        ArrangementAction.get_possible_actions_level_1(
            scene_for_sample=rearrangement1.init_scene
        )
    )
    fig, ax = p_utils.init_3d_figure(name="Rearrangement 1")
    for rearr_action in rearr_actions:
        for rearr_scene in ArrangementAction.get_possible_transitions(
            scene=ArrangementAction.scene_mngr.scene, action=rearr_action
        ):
            # init_scene
            rearrangement1.scene_mngr.render_scene(ax, rearr_scene)
            # rearrangement1.render_axis(rearrangement1.scene_mngr)
            rearrangement1.scene_mngr.show()


if __name__ == "__main__":
    main()
