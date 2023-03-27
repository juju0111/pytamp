from pykin.utils.mesh_utils import get_object_mesh
import h5py, json, os
import trimesh

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
        cls, object_names, object_meshes, support_mesh, distance_above_support=0.002, gaussian=None
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
            )
        return s


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