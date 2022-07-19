from pykin.utils import plot_utils as p_utils
from pytamp.action.pick import PickAction
from pytamp.action.place import PlaceAction
from pytamp.benchmark import Benchmark3

benchmark3 = Benchmark3(robot_name="doosan", geom="collision", is_pyplot=True)
pick = PickAction(benchmark3.scene_mngr, n_contacts=3, n_directions=5, retreat_distance=0.1)
place = PlaceAction(benchmark3.scene_mngr, n_samples_held_obj=0, n_samples_support_obj=10)

################# Action Test ##################

for pick_obj in ["arch_box", "can", "rect_box", "half_cylinder_box", "square_box"]:
    fig, ax = p_utils.init_3d_figure(name="Level wise 1")
    for place_obj in ["clearbox_1_8", "clearbox_1_16", "table"]:
        pick_action = pick.get_action_level_1_for_single_object(pick.scene_mngr.init_scene, pick_obj)
        for grasp_pose in pick_action[pick.info.GRASP_POSES]:
            pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_grasp])
        for pick_scene in pick.get_possible_transitions(pick.scene_mngr.init_scene, pick_action):
            place_action = place.get_action_level_1_for_single_object(place_obj, pick_obj, pick_scene.robot.gripper.grasp_pose, scene=pick_scene)
            for release_pose, obj_pose in place_action[place.info.RELEASE_POSES]:
                place.scene_mngr.render.render_axis(ax, release_pose[place.move_data.MOVE_release])
                place.scene_mngr.render.render_object(ax, place.scene_mngr.scene.objs[pick_obj], obj_pose)
            
    place.scene_mngr.render_objects(ax, alpha=0.1)
    p_utils.plot_basis(ax)
    place.show()
