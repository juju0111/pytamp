from pytamp.benchmark import Rearrange1
from pytamp.utils.making_scene_utils import load_mesh, get_obj_name, Make_Scene
from pykin.utils.mesh_utils import get_object_mesh, get_object_mesh_acronym
from pykin.utils import plot_utils as p_utils

import easydict, os 


def make_scene():
    home_path = os.path.join(os.path.expanduser('~'), 'contact_graspnet/acronym/')
    def custom_parser():
        # object는 parser.add_argument( ~ , nargs="+") , nargs="+" 때문에 list로 arg 셋팅함
        args = easydict.EasyDict(
            {
                "objects": [
                    # "/home/juju/contact_graspnet/acronym/grasps/Candle_b94fcdffbd1befa57f5e345e9a3e5d44_0.012740999337464653.h5",
                    # "/home/juju/contact_graspnet/acronym/grasps/Canister_714320da4aafcb4a47be2353d2b2403b_0.00023318612778400807.h5",
                    # "/home/juju/contact_graspnet/acronym/grasps/Bowl_2efc35a3625fa50961a9876fa6384765_0.012449533111417973.h5",
                    # "/home/juju/contact_graspnet/acronym/grasps/Xbox360_435f39e98d2260f0d6e21b8525c3f8bb_0.002061950217848804.h5"
                ],
                "support": home_path + "grasps/3Shelves_29b66fc9db2f1558e0e89fd83955713c_0.0025867867973150068.h5",
                "num_grasps": 5,
                "mesh_root": home_path,
                "support_scale": 0.025,
            }
        )
        return args

    obj_dict = {}

    args = custom_parser()

    args.objects.append("ben_cube.stl")
    args.objects.append("bottle.stl")
    args.objects.append("bottle.stl")

    obj_dict = {}
    object_meshes = []
    object_names = []

    for o in args.objects:
        if ".h5" in o:
            object_meshes.append(load_mesh(o, mesh_root_dir=args.mesh_root))
            object_names.append(get_obj_name(obj_dict, o))
        if ".stl" in o:
            if "cube" in o:
                object_meshes.append(get_object_mesh(o, 0.075))
                object_names.append(get_obj_name(obj_dict, o))
            else:
                object_meshes.append(get_object_mesh(o))
                object_names.append(get_obj_name(obj_dict, o))

    # for PYTAMP
    support_mesh = get_object_mesh("ben_table.stl", scale=[1.0, 1.5, 1.0])
    init_scene = Make_Scene.random_arrangement(
        object_names, object_meshes, support_mesh
    )
    goal_scene = Make_Scene.random_arrangement(
        object_names, object_meshes, support_mesh, for_goal_scene=True
    )

    rearrangement_scene = Rearrange1(
        "doosan", object_names, init_scene, goal_scene, is_pyplot=False
    )

    # fig, ax = p_utils.init_3d_figure(name="Rearrangement 1")

    # # init_scene
    # rearrangement_scene.scene_mngr.render_scene(ax)
    # rearrangement_scene.render_axis(rearrangement_scene.scene_mngr)
    # rearrangement_scene.scene_mngr.show()

    # # goal_scene
    # rearrangement_scene.goal_scene_mngr.render_scene(ax)
    # rearrangement_scene.render_axis(rearrangement_scene.goal_scene_mngr)
    # rearrangement_scene.goal_scene_mngr.show()

    return rearrangement_scene


if __name__ == "__main__":
    make_scene()
