import trimesh 
import numpy as np
from copy import deepcopy

from pykin.utils import mesh_utils as m_utils
from pytamp.utils.making_scene_utils import Make_Scene
from pytamp.scene.scene import Scene


def get_obj_point_clouds(sample_scene:Make_Scene, scene:Scene, manipulate_obj_name):
    point_clouds = trimesh.PointCloud(np.zeros((1,3))).vertices
    pc_count = 0
    pass_count = 0
    for _, item in enumerate(scene.objs.items()):
        # The support object only considers the top plane
        # So there is no need for point clouds in other parts of the mesh except for the top plane.
        name, i = item

        if manipulate_obj_name == name:
            pc_count = _

        if name in sample_scene._support_objects.keys():
            pass_count += 1
            continue

        copied_mesh = deepcopy(i.gparam)
        copied_mesh.apply_translation(-i.gparam.center_mass)
        copied_mesh.apply_transform(i.h_mat)

        # random sampling으로 mesh 위 point cloud 일부 가져오기 
        points = copied_mesh.sample(1000)
        point_clouds = np.vstack([point_clouds, points])
        # print(name, len(point_clouds))

    return point_clouds[1:], pc_count-pass_count

def get_support_space_point_cloud(sample_scene:Make_Scene, scene:Scene):
    # consider table top point cloud..! 
    # In the case of drawers and bookshelves, it is a little lacking to consider
    support_polys, support_T, sup_obj_name = sample_scene._get_support_polygons()

    support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]

    pts = trimesh.path.polygons.sample(
                        support_polys[support_index], count=3000
                    )
    z_arr = np.full((len(pts), 1), 0)
    o_arr = np.full((len(pts), 1), 1)

    sup_point_cloud = np.hstack((pts, z_arr))
    sup_point_cloud = np.hstack((sup_point_cloud, o_arr))

    transformed_point_cloud = np.dot(support_T[support_index][:3], 
                                        sup_point_cloud.T).T + scene.objs[sup_obj_name[support_index]].h_mat[:3,3]

    return transformed_point_cloud

def get_combined_point_cloud(current_scene:Scene, next_scene:Scene, obj_name, c_pc, n_pc,count,):
    # Transformation matrix from current_scene obj_pose to next_scene obj_pose
    cTn = m_utils.get_relative_transform(current_scene.objs[obj_name].h_mat,
                                            next_scene.objs[obj_name].h_mat)
    
    o_arr = np.full((len(c_pc), 1), 1)
    c_pc_h = np.hstack((c_pc, o_arr))

    transformed_pc =  np.dot(c_pc_h, cTn[:3].T)
    
    # obj mean x,y position
    current_mean = np.array([np.mean(transformed_pc[1000*count:1000*(count+1)][:,0]), 
                            np.mean(transformed_pc[1000*count:1000*(count+1)][:,1]),
                            np.mean(transformed_pc[1000*count:1000*(count+1)][:,2])])
    next_mean = np.array([np.mean(n_pc[1000*count:1000*(count+1)][:,0]), 
                            np.mean(n_pc[1000*count:1000*(count+1)][:,1]),
                            np.mean(n_pc[1000*count:1000*(count+1)][:,2])])
    
    transition_xy = next_mean - current_mean

    transformed_current_pc = transformed_pc + transition_xy

    combined_pc = np.vstack([n_pc, transformed_current_pc])
    return combined_pc