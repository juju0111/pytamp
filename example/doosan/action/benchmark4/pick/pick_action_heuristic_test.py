from pykin.utils import plot_utils as p_utils
from pytamp.action.pick import PickAction
from pytamp.benchmark import Benchmark4

benchmark4 = Benchmark4(robot_name="doosan", geom="visual", is_pyplot=False, disk_num=6)
pick = PickAction(benchmark4.scene_mngr, n_contacts=0, n_directions=0, retreat_distance=0.1)

################# Action Test ##################
fig, ax = p_utils.init_3d_figure(name="Heuristic")
for obj in ["hanoi_disk_0", "hanoi_disk_1", "hanoi_disk_2", "hanoi_disk_3", "hanoi_disk_4", "hanoi_disk_5"]:
    print(obj)
    pose = list(pick.get_grasp_pose_from_heuristic(obj_name=obj))
    for i in range(len(pose)):
        pick.scene_mngr.set_gripper_pose(pose[i][pick.move_data.MOVE_grasp])
        pick.scene_mngr.render_gripper(ax)
        pick.scene_mngr.render_objects(ax)
    
p_utils.plot_basis(ax)
pick.show()
