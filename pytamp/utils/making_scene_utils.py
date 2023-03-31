from pykin.utils.mesh_utils import get_object_mesh
import h5py, json, os
import trimesh
import trimesh.transformations as tra

from acronym_tools import Scene as Scene_ACRONYM
import numpy as np

class Make_Scene(Scene_ACRONYM):
    obj_dict = {}
    object_meshes = []
    object_naems = []
    def __init__(self):
        super().__init__()

    def get_obj_name(self,obj_fname):
        # set mesh_name for Scene
        if obj_fname.endswith(".h5"):
            obj_name = [i for i in obj_fname.split('/') if 'h5' in i]
            obj_name = obj_name[0].split('_')[0]
        elif obj_fname.endswith(".stl"):
            obj_name = obj_fname.split('.')[0]

        if obj_name not in self.obj_dict.keys():
            self.obj_dict[obj_name] = 0
            obj_name = obj_name + str(self.obj_dict[obj_name])
        else:
            self.obj_dict[obj_name] += 1
            obj_name = obj_name + str(self.obj_dict[obj_name])
        return obj_name
    
    @classmethod
    def random_arrangement(
        cls, object_names, object_meshes, support_mesh, distance_above_support=0.002, gaussian=None, for_goal_scene=False
    ):
        """Generate a random scene by arranging all object meshes on any support surface of a provided support mesh.

        Args:
            object_names (list[str]): List of names name corresponding to the meshes
            object_meshes (list[trimesh.Trimesh]): List of meshes of all objects to be placed on top of the support mesh.
            support_mesh (trimesh.Trimesh): Mesh of the support object.
            distance_above_support (float, optional): Distance the object mesh will be placed above the support surface. Defaults to 0.0.
            gaussian (list[float], optional): Normal distribution for position in plane (mean_x, mean_y, std_x, std_y). Defaults to None.

        Returns:
            Scene: Scene representation.
        """
        s = cls()
        s.add_object("support_object", support_mesh, pose=np.eye(4), support=True)

        for i, obj_mesh in enumerate(object_meshes):
            s.place_object(
                object_names[i],
                obj_mesh,
                distance_above_support=distance_above_support,
                gaussian=gaussian,
                for_goal_scene =for_goal_scene
            )
        return s

    def _get_random_stable_pose(self, stable_poses, stable_poses_probs):
        """Return a stable pose according to their likelihood.

        Args:
            stable_poses (list[np.ndarray]): List of stable poses as 4x4 matrices.
            stable_poses_probs (list[float]): List of probabilities.

        Returns:
            np.ndarray: homogeneous 4x4 matrix
        """
        index = np.random.choice(len(stable_poses), p=stable_poses_probs)
        inplane_rot = tra.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stable_poses[index])
    
    def _get_random_stable_pose_for_goal_scene(self, stable_poses, stable_poses_probs):
        """Return a stable pose according to their likelihood.

        Args:
            stable_poses (list[np.ndarray]): List of stable poses as 4x4 matrices.
            stable_poses_probs (list[float]): List of probabilities.

        Returns:
            np.ndarray: homogeneous 4x4 matrix (z axis == 1)
        """
        # index = np.random.choice(len(stable_poses), p=stable_poses_probs)
        ## index : only returm z axis == 1 
        index = np.where(stable_poses[:,2,2] > 0.98)[0][0]
        inplane_rot = tra.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stable_poses[index])
    
    def find_object_placement(
        self, obj_mesh, max_iter, distance_above_support, gaussian=None, for_goal_scene=False,
    ):
        """Try to find a non-colliding stable pose on top of any support surface.

        Args:
            obj_mesh (trimesh.Trimesh): Object mesh to be placed.
            max_iter (int): Maximum number of attempts to place to object randomly.
            distance_above_support (float): Distance the object mesh will be placed above the support surface.
            gaussian (list[float], optional): Normal distribution for position in plane (mean_x, mean_y, std_x, std_y). Defaults to None.

        Raises:
            RuntimeError: In case the support object(s) do not provide any support surfaces.

        Returns:
            bool: Whether a placement pose was found.
            np.ndarray: Homogenous 4x4 matrix describing the object placement pose. Or None if none was found.
        """
        support_polys, support_T = self._get_support_polygons()
        if len(support_polys) == 0:
            raise RuntimeError("No support polygons found!")

        # get stable poses for object
        stable_obj = obj_mesh.copy()
        stable_obj.vertices -= stable_obj.center_mass
        stable_poses, stable_poses_probs = stable_obj.compute_stable_poses(
            threshold=0, sigma=0, n_samples=1
        )
        # stable_poses, stable_poses_probs = obj_mesh.compute_stable_poses(threshold=0, sigma=0, n_samples=1)

        # Sample support index
        support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]

        iter = 0
        colliding = True
        while iter < max_iter and colliding:

            # Sample position in plane
            if gaussian:
                while True:
                    p = Point(
                        np.random.normal(
                            loc=np.array(gaussian[:2])
                            + support_polys[support_index].centroid,
                            scale=gaussian[2:],
                        )
                    )
                    if p.within(support_polys[support_index]):
                        pts = [p.x, p.y]
                        break
            else:
                pts = trimesh.path.polygons.sample(
                    support_polys[support_index], count=1
                )

            # To avoid collisions with the support surface
            pts3d = np.append(pts, distance_above_support)

            # Transform plane coordinates into scene coordinates
            placement_T = np.dot(
                support_T[support_index],
                trimesh.transformations.translation_matrix(pts3d),
            )

            if for_goal_scene:
                pose = self._get_random_stable_pose_for_goal_scene(stable_poses, stable_poses_probs)
            else:
                pose = self._get_random_stable_pose(stable_poses, stable_poses_probs)

            placement_T = np.dot(
                np.dot(placement_T, pose), tra.translation_matrix(-obj_mesh.center_mass)
            )
            # placement_T = np.dot(placement_T, pose)

            # Check collisions
            colliding = self.in_collision_with(
                obj_mesh, placement_T, min_distance=distance_above_support
            )

            iter += 1

        return not colliding, placement_T if not colliding else None

    def place_object(
        self, obj_id, obj_mesh, max_iter=100, distance_above_support=0.0, gaussian=None, for_goal_scene=False,
    ):
        """Add object and place it in a non-colliding stable pose on top of any support surface.

        Args:
            obj_id (str): Name of the object to place.
            obj_mesh (trimesh.Trimesh): Mesh of the object to be placed.
            max_iter (int, optional): Maximum number of attempts to find a placement pose. Defaults to 100.
            distance_above_support (float, optional): Distance the object mesh will be placed above the support surface. Defaults to 0.0.
            gaussian (list[float], optional): Normal distribution for position in plane (mean_x, mean_y, std_x, std_y). Defaults to None.

        Returns:
            [type]: [description]
        """
        success, placement_T = self.find_object_placement(
            obj_mesh,
            max_iter,
            distance_above_support=distance_above_support,
            gaussian=gaussian,
            for_goal_scene=for_goal_scene,
        )

        if success:
            self.add_object(obj_id, obj_mesh, placement_T)
        else:
            print("Couldn't place object", obj_id, "!")

        return success

def load_mesh(filename, mesh_root_dir, scale=None):
    """Load a mesh from a JSON or HDF5 file from the grasp dataset. The mesh will be scaled accordingly.

    This function is for ACRONYM Mesh

    Args:
        filename (str): JSON or HDF5 file name.
        scale (float, optional): If specified, use this as scale instead of value from the file. Defaults to None.

    Returns:
        trimesh.Trimesh: Mesh of the loaded object.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        mesh_fname = data["object"].decode('utf-8')
        mesh_scale = data["object_scale"] if scale is None else scale
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()] if scale is None else scale
    else:
        raise RuntimeError("Unknown file ending:", filename)

    obj_mesh = trimesh.load(os.path.join(mesh_root_dir, mesh_fname))
    obj_mesh = obj_mesh.apply_scale(mesh_scale)

    return obj_mesh

def load_mesh_stl(filename, scale):
    """
    This function is for .stl mesh
    """
    return get_object_mesh(filename, scale)
    

def get_obj_name(obj_dict, obj_fname):
    
    if obj_fname.endswith(".h5"):
        obj_name = [i for i in obj_fname.split('/') if 'h5' in i]
        obj_name = obj_name[0].split('_')[0]
    elif obj_fname.endswith(".stl"):
        obj_name = obj_fname.split('.')[0]
    
    if obj_name not in obj_dict.keys():
        obj_dict[obj_name] = 0
        obj_name = obj_name + str(obj_dict[obj_name])
    else:
        obj_dict[obj_name] += 1
        obj_name = obj_name + str(obj_dict[obj_name])
    return obj_name