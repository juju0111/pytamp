import trimesh
import numpy as np
from copy import deepcopy

from pykin.utils import mesh_utils as m_utils
from pytamp.utils.making_scene_utils import Make_Scene
from pytamp.scene.scene import Scene

def get_obj_point_clouds(sample_scene: Make_Scene, scene: Scene, manipulate_obj_name):
    pc_full = trimesh.PointCloud(np.zeros((1, 3))).vertices
    pc_segments = {}
    pc_colors = np.ones((1, 3), dtype=np.uint8)
    pc_count = 0
    pass_count = 0

    sample_num = 3000
    for _, item in enumerate(scene.objs.items()):
        # The support object only considers the top plane
        # So there is no need for point clouds in other parts of the mesh except for the top plane.
        name, i = item
        pass_bool = False
        for k in sample_scene._support_objects.keys():
            if k in name:
                pass_count += 1
                pass_bool = True
                continue
        if pass_bool:
            continue

        copied_mesh = deepcopy(i.gparam)
        copied_mesh.apply_translation(-i.gparam.center_mass)
        copied_mesh.apply_transform(i.h_mat)

        # random sampling으로 mesh 위 point cloud 일부 가져오기
        points = copied_mesh.sample(sample_num).astype(
            np.float32
        )  # how many PC do you want to sample! you can change number.
        if manipulate_obj_name == name:
            pc_segments[manipulate_obj_name] = np.array(points, dtype=np.float32)
            pc_count = _

        color = (np.ones((sample_num, 3)) * scene.objs[name].color).astype(np.uint8)
        pc_full = np.vstack([pc_full, points])
        pc_colors = np.vstack([pc_colors, color])

        # print(name, len(point_clouds))

    return pc_full[1:], pc_segments, pc_colors[1:], pc_count - pass_count


def get_support_space_point_cloud(sample_scene: Make_Scene, scene: Scene):
    # consider table top point cloud..!
    # In the case of drawers and bookshelves, it is a little lacking to consider
    pc_colors = np.ones((1, 3), dtype=np.uint8)
    sample_num = 3000

    support_polys, support_T, sup_obj_name = sample_scene._get_support_polygons()
    # print("sup_obj :", sup_obj_name)
    support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]

    pts = trimesh.path.polygons.sample(support_polys[support_index], count=sample_num)
    z_arr = np.full((len(pts), 1), 0)
    o_arr = np.full((len(pts), 1), 1)

    sup_point_cloud = np.hstack((pts, z_arr))
    sup_point_cloud = np.hstack((sup_point_cloud, o_arr))

    transformed_point_cloud = (
        np.dot(support_T[support_index][:3], sup_point_cloud.T).T
        + scene.objs[sup_obj_name[support_index]].h_mat[:3, 3]
    )

    color = (np.ones((sample_num, 3)) * scene.objs[sup_obj_name[support_index]].color).astype(
        np.int8
    )
    pc_colors = np.vstack([pc_colors, color])

    return transformed_point_cloud, pc_colors[1:]


def get_combined_point_cloud(
    current_scene: Scene,
    next_scene: Scene,
    obj_name,
    c_pc,
    n_pc,
    count,
):
    sample_num = 3000

    # Transformation matrix from current_scene obj_pose to next_scene obj_pose
    cTn = m_utils.get_relative_transform(
        current_scene.objs[obj_name].h_mat, next_scene.objs[obj_name].h_mat
    )

    o_arr = np.full((len(c_pc), 1), 1)
    c_pc_h = np.hstack((c_pc, o_arr))  # current pc homogenous matrix

    # from c_pc to n_pc
    transformed_pc = np.dot(c_pc_h, cTn[:3].T)

    # obj mean x,y position
    current_mean = np.array(
        [
            np.mean(transformed_pc[sample_num * count : sample_num * (count + 1)][:, 0]),
            np.mean(transformed_pc[sample_num * count : sample_num * (count + 1)][:, 1]),
            np.mean(transformed_pc[sample_num * count : sample_num * (count + 1)][:, 2]),
        ]
    )
    next_mean = np.array(
        [
            np.mean(n_pc[sample_num * count : sample_num * (count + 1)][:, 0]),
            np.mean(n_pc[sample_num * count : sample_num * (count + 1)][:, 1]),
            np.mean(n_pc[sample_num * count : sample_num * (count + 1)][:, 2]),
        ]
    )

    # shift xyz on current_pc
    transition_xy = next_mean - current_mean

    transformed_current_pc = transformed_pc + transition_xy

    combined_pc = np.vstack([n_pc, transformed_current_pc])
    return combined_pc, transition_xy, cTn


def get_mixed_scene(rearr_action, next_scene, current_scene, obj_to_manipulate):
    rearr_action.deepcopy_scene(next_scene)

    for name, obj in next_scene.objs.items():
        rearr_action.scene_mngr.set_object_pose(name, obj.h_mat)

    currnent_obj_pose = deepcopy(current_scene.objs[obj_to_manipulate].h_mat)
    transformed_h_mat = np.eye(4)
    for name, obj in current_scene.objs.items():
        name_ = name + "_current"
        if name == obj_to_manipulate:
            rel_T = m_utils.get_relative_transform(
                current_scene.objs[obj_to_manipulate].h_mat,
                next_scene.objs[obj_to_manipulate].h_mat,
            )
            transformed_h_mat = deepcopy(obj.h_mat) @ rel_T

        else:
            rel_T = m_utils.get_relative_transform(currnent_obj_pose, obj.h_mat)

            transformed_h_mat = deepcopy(next_scene.objs[obj_to_manipulate].h_mat) @ rel_T

        rearr_action.scene_mngr.add_object(
            name_, obj.gtype, obj.gparam, transformed_h_mat, obj.color - 3
        )
    return rearr_action

