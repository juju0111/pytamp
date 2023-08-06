from pykin.utils.mesh_utils import get_object_mesh
import h5py, json, os
import trimesh
import trimesh.transformations as tra
from shapely.geometry import Point

from acronym_tools import Scene as Scene_ACRONYM
import numpy as np
import pyrender


class Make_Scene(Scene_ACRONYM):
    obj_dict = {}
    object_meshes = []
    object_naems = []

    def __init__(
        self,
    ):
        super().__init__()

    def get_obj_name(self, obj_fname):
        # set mesh_name for Scene
        if obj_fname.endswith(".h5"):
            obj_name = [i for i in obj_fname.split("/") if "h5" in i]
            obj_name = obj_name[0].split("_")[0]
        elif obj_fname.endswith(".stl"):
            obj_name = obj_fname.split(".")[0]

        if obj_name not in self.obj_dict.keys():
            self.obj_dict[obj_name] = 0
            obj_name = obj_name + str(self.obj_dict[obj_name])
        else:
            self.obj_dict[obj_name] += 1
            obj_name = obj_name + str(self.obj_dict[obj_name])
        return obj_name

    @classmethod
    def random_arrangement(
        cls,
        object_names,
        object_meshes,
        support_names,
        support_meshes,
        base_mesh="table",
        distance_above_support=0.002,
        gaussian=None,
        for_goal_scene=False,
        use_distance_limit=True,
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
        for n, m in zip(support_names, support_meshes):
            s.add_object(n, m, pose=np.eye(4), support=True)

        for i, obj_mesh in enumerate(object_meshes):
            s.place_object(
                object_names[i],
                obj_mesh,
                distance_above_support=distance_above_support,
                gaussian=gaussian,
                for_goal_scene=for_goal_scene,
                base_mesh=base_mesh,
                use_distance_limit=use_distance_limit,
            )
        return s

    @classmethod
    def static_arrangement(
        cls,
        object_names,
        object_meshes,
        object_poses,
        support_names,
        support_meshes,
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

        for n, m in zip(support_names, support_meshes):
            s.add_object(n, m, pose=np.eye(4), support=True)

        for i, obj_mesh in enumerate(object_meshes):
            s.place_object_specific_pose(object_names[i], obj_mesh, object_poses[i])
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
        index = np.where(stable_poses[:, 2, 2] > 0.98)[0][0]
        inplane_rot = tra.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stable_poses[index])

    def find_object_placement(
        self,
        obj_mesh,
        max_iter,
        distance_above_support,
        support_obj_name=None,
        gaussian=None,
        for_goal_scene=False,
        use_distance_limit=True,
        erosion_dist=0.015,
    ):
        """Try to find a non-colliding stable pose on top of any support surface.

        Args:
            support_obj_name (str) : Support obj name to place.
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

        support_polys, support_T, sup_obj_names = self._get_specific_support_polygons(
            support_obj_name, erosion_distance=erosion_dist
        )
        if len(support_polys) == 0:
            raise RuntimeError("No support polygons found!")
        # get stable poses for object
        stable_obj = obj_mesh.copy()
        stable_obj.vertices -= stable_obj.center_mass
        stable_poses, stable_poses_probs = stable_obj.compute_stable_poses(
            threshold=0, sigma=0, n_samples=1
        )
        # stable_poses, stable_poses_probs = obj_mesh.compute_stable_poses(threshold=0, sigma=0, n_samples=1)
        # print(support_obj_name)
        # Sample support index
        if support_obj_name == "table":
            support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]
        elif support_obj_name == "shelves":
            support_index = 7
            # support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]
        iter = 0
        colliding = True
        while iter < max_iter and colliding:
            # Sample position in plane
            if gaussian:
                while True:
                    p = Point(
                        np.random.normal(
                            loc=np.array(gaussian[:2])
                            + np.array(
                                [
                                    support_polys[support_index].centroid.x,
                                    support_polys[support_index].centroid.y,
                                ]
                            ),
                            scale=gaussian[2:],
                        )
                    )
                    # print("sampled distance : ", ((-p.x - .9)**2 + (p.y + 0.6)**2) )
                    if use_distance_limit:
                        if ((-p.x - 1.0) ** 2 + (p.y + 0.6) ** 2) > 0.7:
                            continue
                    if p.within(support_polys[support_index]):
                        pts = [p.x, p.y]
                        break
            else:
                while True:
                    pts = trimesh.path.polygons.sample(support_polys[support_index], count=1)

                    # for robot arm reach
                    # print(pts)
                    if use_distance_limit:
                        if ((-pts[0][0] - 1.0) ** 2 + (pts[0][1] + 0.6) ** 2) < 0.4:
                            break
                    else:
                        break

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

        return not colliding, placement_T, sup_obj_names[support_index] if not colliding else None

    def place_object(
        self,
        obj_id,
        obj_mesh,
        max_iter=100,
        distance_above_support=0.0,
        gaussian=None,
        for_goal_scene=False,
        base_mesh="table",
        use_distance_limit=True,
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
        success, placement_T, sup_obj_name = self.find_object_placement(
            obj_mesh,
            max_iter,
            distance_above_support=distance_above_support,
            support_obj_name=base_mesh,
            gaussian=gaussian,
            for_goal_scene=for_goal_scene,
            use_distance_limit=use_distance_limit,
        )

        if success:
            self.add_object(obj_id, obj_mesh, placement_T)
        else:
            print("Couldn't place object", obj_id, "!")

        return success

    def find_object_placement_from_specific_object(
        self,
        obj_mesh,
        distance_above_support,
        support_obj_name,
        for_goal_scene=False,
        erosion_dist=0.015,
    ):
        """Try to find a non-colliding stable pose on top of any support surface.

        Args:
            support_obj_name (str) : Support obj name to place.
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
        support_polys, support_T, sup_obj_names = self._get_specific_support_polygons(
            support_obj_name, erosion_distance=erosion_dist
        )

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

        pts = trimesh.path.polygons.sample(support_polys[support_index], count=1)

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

        # Check collisions pass
        # colliding = self.in_collision_with(
        #     obj_mesh, placement_T, min_distance=distance_above_support
        # )
        # print(colliding)
        # iter += 1
        return True, placement_T, sup_obj_names[support_index]

    def place_object_specific_support_obect(
        self,
        support_obj_name,
        obj_id,
        obj_mesh,
        max_iter=100,
        distance_above_support=0.0,
        gaussian=None,
        for_goal_scene=False,
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
        success, placement_T, sup_obj_name = self.find_object_placement(
            obj_mesh,
            max_iter,
            distance_above_support=distance_above_support,
            support_obj_name=support_obj_name,
            gaussian=gaussian,
            for_goal_scene=for_goal_scene,
        )

        if success:
            self.add_object(obj_id, obj_mesh, placement_T)
        else:
            print("Couldn't place object", obj_id, "!")

        return success

    def place_object_specific_pose(
        self,
        obj_id,
        obj_mesh,
        placement_T,
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

        self.add_object(obj_id, obj_mesh, placement_T)

    def as_pyrender_scene(self):
        """Return pyrender scene representation.

        Returns:
            pyrender.Scene: Representation of the scene
        """
        pyrender_scene = pyrender.Scene()
        for obj_id, obj_mesh in self._objects.items():
            mesh = pyrender.Mesh.from_trimesh(obj_mesh, smooth=False)
            pyrender_scene.add(mesh, name=obj_id, pose=self._poses[obj_id])
        return pyrender_scene

    def get_support_obj_point_cloud(self):
        # consider table top point cloud..!
        # In the case of drawers and bookshelves, it is a little lacking to consider
        support_polys, support_T, sup_obj_name = self._get_support_polygons()

        support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]

        pts = trimesh.path.polygons.sample(support_polys[support_index], count=3000)
        z_arr = np.full((len(pts), 1), 0)
        o_arr = np.full((len(pts), 1), 1)

        sup_point_cloud = np.hstack((pts, z_arr))
        sup_point_cloud = np.hstack((sup_point_cloud, o_arr))

        transformed_point_cloud = (
            np.dot(support_T[support_index][:3], sup_point_cloud.T).T
            + self.scene_mngr.scene.objs[sup_obj_name[support_index]].h_mat[:3, 3]
        )

        return transformed_point_cloud

    def _get_specific_support_polygons(
        self,
        support_obj_name: str,
        min_area=0.0001,
        gravity=np.array([0, 0, -1.0]),
        erosion_distance=0.015,
    ):
        """Extract support facets by comparing normals with gravity vector and checking area.

        Args:
            support_obj_name
            min_area (float, optional): Minimum area of support facets [m^2]. Defaults to 0.01.
            gravity ([np.ndarray], optional): Gravity vector in scene coordinates. Defaults to np.array([0, 0, -1.0]).
            erosion_distance (float, optional): Clearance from support surface edges. Defaults to 0.02.

        Returns:
            list[trimesh.path.polygons.Polygon]: list of support polygons.
            list[np.ndarray]: list of homogenous 4x4 matrices describing the polygon poses in scene coordinates.
        """
        assert np.isclose(np.linalg.norm(gravity), 1.0)

        support_polygons = []
        support_polygons_T = []
        obj_names = []
        # Add support plane if it is set (although not infinite)
        obj_name = support_obj_name
        # print("object_name : ", obj_name)
        # print("specific sup name :", support_obj_name, self._support_objects)
        if support_obj_name in self._support_objects:
            obj_mesh = self._support_objects.get(support_obj_name)
        else:
            obj_mesh = self._support_objects.get(support_obj_name + "_support")

        # get all facets that are aligned with -gravity and bigger than min_area
        support_facet_indices = np.argsort(obj_mesh.facets_area)
        support_facet_indices = [
            idx
            for idx in support_facet_indices
            if np.isclose(obj_mesh.facets_normal[idx].dot(-gravity), 1.0, atol=0.5)
            and obj_mesh.facets_area[idx] > min_area
        ]

        for inds in support_facet_indices:
            index = inds
            normal = obj_mesh.facets_normal[index]
            origin = obj_mesh.facets_origin[index]

            T = trimesh.geometry.plane_transform(origin, normal)
            vertices = trimesh.transform_points(obj_mesh.vertices, T)[:, :2]

            # find boundary edges for the facet
            edges = obj_mesh.edges_sorted.reshape((-1, 6))[obj_mesh.facets[index]].reshape((-1, 2))
            group = trimesh.grouping.group_rows(edges, require_count=1)

            # run the polygon conversion
            polygon = trimesh.path.polygons.edges_to_polygons(edges=edges[group], vertices=vertices)

            assert len(polygon) == 1

            # erode to avoid object on edges
            polygon[0] = polygon[0].buffer(-erosion_distance)

            if not polygon[0].is_empty and polygon[0].area > min_area:
                support_polygons.append(polygon[0])
                support_polygons_T.append(trimesh.transformations.inverse_matrix(T))
                obj_names.append(obj_name)

        return support_polygons, support_polygons_T, obj_names


class SceneRenderer:
    def __init__(
        self,
        pyrender_scene,
        fov=np.pi / 6.0,
        width=400,
        height=400,
        aspect_ratio=1.0,
        z_near=0.001,
    ):
        """Create an image renderer for a scene.

        Args:
            pyrender_scene (pyrender.Scene): Scene description including object meshes and their poses.
            fov (float, optional): Field of view of camera. Defaults to np.pi/6.
            width (int, optional): Width of camera sensor (in pixels). Defaults to 400.
            height (int, optional): Height of camera sensor (in pixels). Defaults to 400.
            aspect_ratio (float, optional): Aspect ratio of camera sensor. Defaults to 1.0.
            z_near (float, optional): Near plane closer to which nothing is rendered. Defaults to 0.001.
        """
        self._fov = fov
        self._width = width
        self._height = height
        self._z_near = z_near
        self._scene = pyrender_scene

        self._camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=aspect_ratio, znear=z_near)

    def get_trimesh_camera(self):
        """Get a trimesh object representing the camera intrinsics.

        Returns:
            trimesh.scene.cameras.Camera: Intrinsic parameters of the camera model
        """
        return trimesh.scene.cameras.Camera(
            fov=(np.rad2deg(self._fov), np.rad2deg(self._fov)),
            resolution=(self._height, self._width),
            z_near=self._z_near,
        )

    def _to_pointcloud(self, depth):
        """Convert depth image to pointcloud given camera intrinsics.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Point cloud.
        """
        fy = fx = 0.5 / np.tan(self._fov * 0.5)  # aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)

        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T

    def render(self, camera_pose, target_id="", render_pc=True):
        """Render RGB/depth image, point cloud, and segmentation mask of the scene.

        Args:
            camera_pose (np.ndarray): Homogenous 4x4 matrix describing the pose of the camera in scene coordinates.
            target_id (str, optional): Object ID which is used to create the segmentation mask. Defaults to ''.
            render_pc (bool, optional): If true, point cloud is also returned. Defaults to True.

        Returns:
            np.ndarray: Color image.
            np.ndarray: Depth image.
            np.ndarray: Point cloud.
            np.ndarray: Segmentation mask.
        """
        # Keep local to free OpenGl resources after use
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self._width, viewport_height=self._height
        )

        # add camera and light to scene
        scene = self._scene.as_pyrender_scene()
        scene.add(self._camera, pose=camera_pose, name="camera")
        light = pyrender.SpotLight(
            color=np.ones(4),
            intensity=3.0,
            innerConeAngle=np.pi / 16,
            outerConeAngle=np.pi / 6.0,
        )
        scene.add(light, pose=camera_pose, name="light")

        # render the full scene
        color, depth = renderer.render(scene)

        segmentation = np.zeros(depth.shape, dtype=np.uint8)

        # hide all objects
        for node in scene.mesh_nodes:
            node.mesh.is_visible = False

        # Render only target object and add to segmentation mask
        for node in scene.mesh_nodes:
            if node.name == target_id:
                node.mesh.is_visible = True
                _, object_depth = renderer.render(scene)
                mask = np.logical_and((np.abs(object_depth - depth) < 1e-6), np.abs(depth) > 0)
                segmentation[mask] = 1

        for node in scene.mesh_nodes:
            node.mesh.is_visible = True

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None

        return color, depth, pc, segmentation


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
        mesh_fname = data["object"].decode("utf-8")
        mesh_scale = data["object_scale"] if scale is None else scale
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode("utf-8")
        mesh_scale = data["object/scale"][()] if scale is None else scale
    elif filename.endswith(".obj"):
        print("Enter without root address")
        pass
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
        obj_name = [i for i in obj_fname.split("/") if "h5" in i]
        obj_name = obj_name[0].split("_")[0]
    elif obj_fname.endswith(".stl"):
        obj_name = obj_fname.split(".")[0]

    if obj_name not in obj_dict.keys():
        obj_dict[obj_name] = 0
        obj_name = obj_name + str(obj_dict[obj_name])
    else:
        obj_dict[obj_name] += 1
        obj_name = obj_name + str(obj_dict[obj_name])
    return obj_name
