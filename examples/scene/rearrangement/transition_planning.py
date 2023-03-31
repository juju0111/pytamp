from pytamp.benchmark import Rearrange1
from pytamp.utils.making_scene_utils import load_mesh, get_obj_name, Make_Scene
from pykin.utils.mesh_utils import get_object_mesh, get_object_mesh_acronym
from pykin.utils import plot_utils as p_utils

from pytamp.benchmark.rearrange1 import make_scene

def main():
    object_names, init_scene, goal_scene = make_scene()

    rearrangement_scene = Rearrange1('doosan', object_names, init_scene, goal_scene, is_pyplot=False)

    






    fig, ax = p_utils.init_3d_figure(name="Rearrangement 1")

    # init_scene
    rearrangement_scene.scene_mngr.render_scene(ax)
    rearrangement_scene.render_axis(rearrangement_scene.scene_mngr)
    rearrangement_scene.scene_mngr.show()

    # goal_scene
    rearrangement_scene.goal_scene_mngr.render_scene(ax)
    rearrangement_scene.render_axis(rearrangement_scene.goal_scene_mngr)
    rearrangement_scene.goal_scene_mngr.show()



if __name__=="__main__":
    main()