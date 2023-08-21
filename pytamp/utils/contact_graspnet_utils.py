import os
import sys
import argparse
import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import torch.nn.functional as F
import torch
from abc import abstractclassmethod, ABCMeta

from pykin.utils.transform_utils import get_inverse_homogeneous
from pytamp.utils.heuristic_utils import get_heuristic_eef_pose
from pytamp.utils.point_cloud_utils import get_mixed_scene
from pytamp.utils.point_cloud_utils import get_obj_point_clouds
from pytamp.utils.point_cloud_utils import get_support_space_point_cloud


class Grasp_Using_AI_model(metaclass=ABCMeta):
    def __init__(self, action, robot_name, bench_num=0):
        self.action = action
        self.T_cam = np.eye(4)
        self.w_T_cam = np.eye(4)
        self.bench_num = bench_num
        self.robot_name = robot_name

    def reject_median_outliers(self, data, m=0.4, z_only=False):
        """
        Reject outliers with median absolute distance m

        Arguments:
            data {[np.ndarray]} -- Numpy array such as point cloud

        Keyword Arguments:
            m {[float]} -- Maximum absolute distance from median in m (default: {0.4})
            z_only {[bool]} -- filter only via z_component (default: {False})

        Returns:
            [np.ndarray] -- Filtered data without outliers
        """
        if z_only:
            d = np.abs(data[:, 2:3] - np.median(data[:, 2:3]))
        else:
            d = np.abs(data - np.median(data, axis=0, keepdims=True))

        return data[np.sum(d, axis=1) < m]

    def regularize_pc_point_count(self, pc, npoints):
        """
        If point cloud pc has less points than npoints, it oversamples.
        Otherwise, it downsample the input pc to have npoint points.

        :param pc: Nx3 point cloud
        :param npoints: number of points the regularized point cloud should have

        :returns: npointsx3 regularized point cloud
        """

        if pc.shape[0] > npoints:
            print("Random sample points ")
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
            pc = pc[center_indexes, :]
        else:
            required = npoints - pc.shape[0]
            if required > 0:
                index = np.random.choice(range(pc.shape[0]), size=required)
                pc = np.concatenate((pc, pc[index, :]), axis=0)
        return pc

    def extract_3d_cam_boxes(self, full_pc, pc_segments, min_size=0.3, max_size=0.6):
        """
        Extract 3D bounding boxes around the pc_segments for inference to create
        dense and zoomed-in predictions but still take context into account.

        :param full_pc: Nx3 scene point cloud
        :param pc_segments: Mx3 segmented point cloud of the object of interest
        :param min_size: minimum side length of the 3D bounding box
        :param max_size: maximum side length of the 3D bounding box
        :returns: (pc_regions, obj_centers) Point cloud box regions and their centers
        """

        pc_regions = {}
        obj_centers = {}

        for i in pc_segments:
            pc_segments[i] = self.reject_median_outliers(pc_segments[i], m=0.4, z_only=False)

            if np.any(pc_segments[i]):
                max_bounds = np.max(pc_segments[i][:, :3], axis=0)
                min_bounds = np.min(pc_segments[i][:, :3], axis=0)

                obj_extent = max_bounds - min_bounds
                obj_center = min_bounds + obj_extent / 2

                # cube size is between 0.3 and 0.6 depending on object extents
                size = np.minimum(np.maximum(np.max(obj_extent) * 2, min_size), max_size)
                print("Extracted Region Cube Size: ", size)
                partial_pc = full_pc[
                    np.all(full_pc > (obj_center - size / 2), axis=1)
                    & np.all(full_pc < (obj_center + size / 2), axis=1)
                ]
                if np.any(partial_pc):
                    partial_pc = self.regularize_pc_point_count(
                        partial_pc,
                        20000,
                    )
                    pc_regions[i] = partial_pc
                    obj_centers[i] = obj_center

        return pc_regions, obj_centers

    def filter_segment(self, contact_pts, segment_pc, thres=0.00001):
        """
        Filter grasps to obtain contacts on specified point cloud segment

        :param contact_pts: Nx3 contact points of all grasps in the scene
        :param segment_pc: Mx3 segmented point cloud of the object of interest
        :param thres: maximum distance in m of filtered contact points from segmented point cloud
        :returns: Contact/Grasp indices that lie in the point cloud segment
        """
        filtered_grasp_idcs = np.array([], dtype=np.int32)

        if contact_pts.shape[0] > 0 and segment_pc.shape[0] > 0:
            try:
                dists = contact_pts[:, :3].reshape(-1, 1, 3) - segment_pc.reshape(1, -1, 3)
                min_dists = np.min(np.linalg.norm(dists, axis=2), axis=1)
                filtered_grasp_idcs = np.where(min_dists < thres)
            except:
                pass

        return filtered_grasp_idcs

    @abstractclassmethod
    def get_grasp(self):
        raise NotImplementedError


class Grasp_Using_Scale_Balance_GraspNet(Grasp_Using_AI_model):
    def __init__(self, action, robot_name, bench_num=0):
        super().__init__(action, robot_name, bench_num)
        home_path = os.path.expanduser('~')
        self.ROOT_DIR = os.path.dirname(
            home_path + "/scale_balance_grasp_files/Scale-Balanced-Grasp/"
        )
        sys.path.append(os.path.join(self.ROOT_DIR, "models"))
        sys.path.append(os.path.join(self.ROOT_DIR, "utils"))

        cfgs = self.argparse()

        from graspnetAPI import GraspGroup
        from graspnet import GraspNet_MSCQ, pred_decode
        from dsn import DSN, cluster

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = GraspNet_MSCQ(
            input_feature_dim=0,
            num_view=cfgs.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.08,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
            obs=True,
        )
        self.pred_decode = pred_decode
        self.cluster = cluster
        self.GraspGroup = GraspGroup

        checkpoint = torch.load(
            os.path.join(self.ROOT_DIR, cfgs.checkpoint_path), map_location="cpu"
        )
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.to(device)
        self.net = self.net.eval()
        start_epoch = checkpoint["epoch"]
        print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

        # load seg network
        self.seg_net = DSN(input_feature_dim=0)
        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(self.ROOT_DIR, cfgs.seg_checkpoint_path), map_location="cpu"
        )
        self.seg_net.load_state_dict(checkpoint["model_state_dict"])
        self.seg_net.to(device)
        self.seg_net.eval()

        self.T_cam = np.array(
            [
                [6.12323400e-17, 9.51056516e-01, -3.09016994e-01, -4.94508490e-01],
                [-1.00000000e00, 5.82354159e-17, -1.89218337e-17, 6.34369494e-01],
                [-0.00000000e00, 3.09016994e-01, 9.51056516e-01, 1.24082823e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        try:
            w_T_cam = self.action.scene_mngr.scene.objs["table"].h_mat @ self.T_cam
        except:
            w_T_cam = self.action.scene_mngr.scene.objs["shelves"].h_mat @ self.T_cam

        m_ = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.w_T_cam = w_T_cam.dot(m_)

    def argparse(self):
        home_path =  os.path.expanduser("~")

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dataset_root",
            default=home_path + "/scale_balance_grasp_files/Scale-Balanced-Grasp/dataset/graspnet",
            help="Dataset root",
        )
        parser.add_argument(
            "--checkpoint_path",
            default="logs/log_full_model/checkpoint.tar",
            help="Model checkpoint path",
        )
        parser.add_argument(
            "--seg_checkpoint_path",
            default="logs/log_insseg/checkpoint.tar",
            help="Segmentation Model checkpoint path",
        )

        parser.add_argument(
            "--dump_dir", default="logs/dump_full_model", help="Dump dir to save outputs"
        )
        parser.add_argument("--camera", default="realsense", help="Camera split [realsense/kinect]")
        parser.add_argument(
            "--num_point", type=int, default=20000, help="Point Number [default: 20000]"
        )
        parser.add_argument("--num_view", type=int, default=300, help="View Number [default: 300]")
        parser.add_argument(
            "--batch_size", type=int, default=1, help="Batch Size during inference [default: 1]"
        )
        parser.add_argument(
            "--collision_thresh",
            type=float,
            default=0.01,
            help="Collision Threshold in collision detection [default: 0.01]",
        )
        parser.add_argument(
            "--voxel_size",
            type=float,
            default=0.01,
            help="Voxel Size to process point clouds before collision detection [default: 0.01]",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=30,
            help="Number of workers used in evaluation [default: 30]",
        )
        cfgs = parser.parse_args([])
        return cfgs

    def get_grasp_pose_for_scale_balance_grasp(self, rotation_matrices, translations):
        grasp_pose = np.zeros([len(rotation_matrices), 4, 4])
        grasp_pose[:, :3, :3] = rotation_matrices
        grasp_pose[:, :3, 3] = translations
        grasp_pose[:, 3, 3] = 1.0
        return grasp_pose

    def get_pc_from_camera_point_of_view(self, all_pc, pc_segments, obj_to_manipulate):
        self.w_T_cam[:2, 3] = np.mean(pc_segments[obj_to_manipulate], 0)[:2] - [0.05, 0]

        cam_T_w = get_inverse_homogeneous(self.w_T_cam)
        ones_arr = np.full((len(all_pc), 1), 1)
        w_pc = np.hstack((all_pc, ones_arr))

        cam_pc = np.dot(cam_T_w, w_pc.T).T

        # next_pc_segment도 변경해줘
        ones_arr = np.full((len(pc_segments[obj_to_manipulate]), 1), 1)
        w_pc = np.hstack((pc_segments[obj_to_manipulate], ones_arr))

        pc_segments[obj_to_manipulate] = np.dot(cam_T_w, w_pc.T).T

        return cam_pc, pc_segments

    def change_grasp_to_world_coord(self, pred_grasps_cam, obj_to_manipulate):
        def collision_check_using_contact_graspnet(pred_grasps):
            collision_free_grasps = []
            for grasps in pred_grasps:
                self.action.scene_mngr.set_gripper_pose(grasps)
                if not self.action._collide(is_only_gripper=True):
                    collision_free_grasps.append(grasps)

            return np.array(collision_free_grasps)

        pred_grasps_world = {}
        pred_grasps_world_augment = {}
        pred_grasps_cam_augment = {}
        collision_free_grasps = []
        # Z축으로 90도 돌려야함.
        z_90_matrix = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        pred_grasps_world[obj_to_manipulate] = self.w_T_cam @ pred_grasps_cam

        eef_poses = np.zeros((pred_grasps_world[obj_to_manipulate].shape))
        for i in range(len(pred_grasps_world[obj_to_manipulate])):
            eef_poses[i] = self.compute_eef_pose_from_tcp_pose(
                pred_grasps_world[obj_to_manipulate][i], depth=self.gripper_depth[i]
            )

        pred_grasps_world[obj_to_manipulate] = eef_poses

        collision_free_grasps = collision_check_using_contact_graspnet(
            pred_grasps_world[obj_to_manipulate]
        )

        if not len(collision_free_grasps) or self.bench_num == 2:
            pred_grasps_cam_augment[obj_to_manipulate] = pred_grasps_cam @ z_90_matrix
            pred_grasps_world_augment[obj_to_manipulate] = (
                self.w_T_cam @ pred_grasps_cam_augment[obj_to_manipulate]
            )

            collision_free_grasps = collision_check_using_contact_graspnet(
                pred_grasps_world_augment[obj_to_manipulate]
            )

        augmented_grasps = []
        if not len(collision_free_grasps) or self.bench_num == 2:
            pred_grasps_world_augment[obj_to_manipulate] = np.vstack(
                [pred_grasps_world_augment[obj_to_manipulate], pred_grasps_world[obj_to_manipulate]]
            )
            for grasps in pred_grasps_world_augment[obj_to_manipulate]:
                self.action.scene_mngr.set_gripper_pose(grasps)
                tcp_pose = self.action.scene_mngr.scene.robot.gripper.get_gripper_tcp_pose()

                for tcp_pose_ in get_heuristic_eef_pose(tcp_pose):
                    eef_pose_ = (
                        self.action.scene_mngr.scene.robot.gripper.compute_eef_pose_from_tcp_pose(
                            tcp_pose_
                        )
                    )
                    augmented_grasps.append(eef_pose_)

        if augmented_grasps:
            augmented_grasps = np.array(augmented_grasps)
            print("Augment 2 y axis rotation from -pi/3 ~ pi/3 : ", augmented_grasps.shape)
            pred_grasps_world_augment[obj_to_manipulate] = augmented_grasps

            collision_free_grasps = collision_check_using_contact_graspnet(
                pred_grasps_world_augment[obj_to_manipulate]
            )
            print("Collision free grasps step 3 : ", collision_free_grasps.shape)

        return collision_free_grasps

    def compute_eef_pose_from_tcp_pose(self, tcp_pose=np.eye(4), depth=0.01):
        eef_pose = np.eye(4)
        eef_pose[:3, :3] = tcp_pose[:3, :3]
        eef_pose[:3, 3] = tcp_pose[:3, 3] - np.dot(
            abs(self.action.scene_mngr.scene.robot.gripper.tcp_position[-1]) + depth,
            tcp_pose[:3, 2],
        )
        return eef_pose

    def get_grasp(
        self,
        init_scene,
        next_node=None,
        current_node=None,
    ):
        obj_to_manipulate = current_node["action"]["rearr_obj_name"]
        if self.bench_num == 2:
            min_size = 0.6
        else:
            min_size = 0.4

        if next_node != None:
            self.action.get_mixed_scene_on_current(
                next_scene=next_node["state"],
                current_scene=current_node["state"],
                obj_to_manipulate=obj_to_manipulate,
            )
            pc, pc_segments, pc_color, count = get_obj_point_clouds(
                init_scene, self.action.scene_mngr.scene, obj_to_manipulate
            )
        else:
            pc, pc_segments, pc_color, count = get_obj_point_clouds(
                init_scene, current_node["state"], obj_to_manipulate
            )
        table_point_cloud, table_color = get_support_space_point_cloud(
            init_scene, current_node["state"]
        )

        # in pc_utils
        all_pc = np.vstack([pc, table_point_cloud])
        all_color = np.vstack([pc_color, table_color])

        cam_pc, pc_segments = self.get_pc_from_camera_point_of_view(
            all_pc, pc_segments, obj_to_manipulate
        )

        pc_regions, _ = self.extract_3d_cam_boxes(cam_pc[:, :3], pc_segments)

        data = {}
        # data['point_clouds'] = torch.unsqueeze(torch.from_numpy(pc_sampled),0).cuda().to(torch.float32)
        data["point_clouds"] = (
            torch.unsqueeze(torch.from_numpy(pc_regions[obj_to_manipulate]), 0)
            .cuda()
            .to(torch.float32)
        )

        with torch.no_grad():
            end_points = self.seg_net(data)
            batch_xyz_img = end_points["point_clouds"]
            B, _, N = batch_xyz_img.shape
            batch_offsets = end_points["center_offsets"]
            batch_fg = end_points["foreground_logits"]
            batch_fg = F.softmax(batch_fg, dim=1)
            batch_fg = torch.argmax(batch_fg, dim=1)
            # end_points["instance_mask"] = batch_fg
            clustered_imgs = []
            for i in range(B):
                clustered_img, uniq_cluster_centers = self.cluster(
                    batch_xyz_img[i], batch_offsets[i].permute(1, 0), batch_fg[i]
                )
                clustered_img = clustered_img.unsqueeze(0)
                clustered_imgs.append(clustered_img)
            end_points["seed_cluster"] = torch.cat(clustered_imgs, dim=0)
            end_points = self.net(data)
            grasp_preds = self.pred_decode(end_points)

        gg_array = grasp_preds[0].detach().cpu().numpy()

        segments_inds = self.filter_segment(
            gg_array[:, 13:16], pc_segments[obj_to_manipulate], thres=0.001
        )
        grasp_to_manipulate = gg_array[segments_inds]
        grasp_thresh = grasp_to_manipulate[grasp_to_manipulate[:, 0] > 0.3]
        grasp_thresh = grasp_thresh[grasp_thresh[:, 12] > 0]

        gg = self.GraspGroup(grasp_thresh)

        gripper_tcp_point = self.get_grasp_pose_for_scale_balance_grasp(
            gg.rotation_matrices, gg.translations
        )
        self.gripper_depth = gg.depths

        collision_free_grasps = self.change_grasp_to_world_coord(
            gripper_tcp_point, obj_to_manipulate
        )

        return collision_free_grasps


class Grasp_Using_FGC_GraspNet(Grasp_Using_AI_model):
    def __init__(self, action, robot_name, bench_num=0):
        super().__init__(action, robot_name, bench_num)
        home_path = os.path.expanduser("~")
        self.ROOT_DIR = os.path.dirname(home_path + "/scale_balance_grasp_files/FGC-GraspNet/")
        sys.path.append(os.path.join(self.ROOT_DIR, "models"))
        sys.path.append(os.path.join(self.ROOT_DIR, "utils"))
        print(sys.path)

        cfgs = self.argparse()

        from graspnetAPI import GraspGroup
        from FGC_graspnet import FGC_graspnet
        from decode import pred_decode

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = FGC_graspnet(
            input_feature_dim=0,
            num_view=cfgs.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax=0.02,
            is_training=False,
            is_demo=True,
        )

        self.pred_decode = pred_decode
        self.GraspGroup = GraspGroup

        checkpoint = torch.load(cfgs.checkpoint_path)

        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.to(device)
        self.net = self.net.eval()
        start_epoch = checkpoint["epoch"]
        print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

        self.T_cam = np.array(
            [
                [6.12323400e-17, 9.51056516e-01, -3.09016994e-01, -4.94508490e-01],
                [-1.00000000e00, 5.82354159e-17, -1.89218337e-17, 6.34369494e-01],
                [-0.00000000e00, 3.09016994e-01, 9.51056516e-01, 1.24082823e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        try:
            w_T_cam = self.action.scene_mngr.scene.objs["table"].h_mat @ self.T_cam
        except:
            w_T_cam = self.action.scene_mngr.scene.objs["shelves"].h_mat @ self.T_cam

        m_ = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.w_T_cam = w_T_cam.dot(m_)

    def argparse(self):
        home_path = os.path.expanduser("~")
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--checkpoint_path",
            default=home_path + "/scale_balance_grasp_files/FGC-GraspNet/log/realsense_checkpoint.tar",
            help="Model checkpoint path",
        )
        parser.add_argument(
            "--num_point", type=int, default=20000, help="Point Number [default: 20000]"
        )
        parser.add_argument("--num_view", type=int, default=300, help="View Number [default: 300]")
        parser.add_argument(
            "--collision_thresh",
            type=float,
            default=0.01,
            help="Collision Threshold in collision detection [default: 0.01]",
        )
        parser.add_argument(
            "--voxel_size",
            type=float,
            default=0.01,
            help="Voxel Size to process point clouds before collision detection [default: 0.01]",
        )
        cfgs = parser.parse_args([])
        return cfgs

    def get_grasp_pose_for_scale_balance_grasp(self, rotation_matrices, translations):
        grasp_pose = np.zeros([len(rotation_matrices), 4, 4])
        grasp_pose[:, :3, :3] = rotation_matrices
        grasp_pose[:, :3, 3] = translations
        grasp_pose[:, 3, 3] = 1.0
        return grasp_pose

    def get_pc_from_camera_point_of_view(self, all_pc, pc_segments, obj_to_manipulate):
        self.w_T_cam[:2, 3] = np.mean(pc_segments[obj_to_manipulate], 0)[:2] - [0.05, 0]

        cam_T_w = get_inverse_homogeneous(self.w_T_cam)
        ones_arr = np.full((len(all_pc), 1), 1)
        w_pc = np.hstack((all_pc, ones_arr))

        cam_pc = np.dot(cam_T_w, w_pc.T).T

        # next_pc_segment도 변경해줘
        ones_arr = np.full((len(pc_segments[obj_to_manipulate]), 1), 1)
        w_pc = np.hstack((pc_segments[obj_to_manipulate], ones_arr))

        pc_segments[obj_to_manipulate] = np.dot(cam_T_w, w_pc.T).T

        return cam_pc, pc_segments

    def change_grasp_to_world_coord(self, pred_grasps_cam, obj_to_manipulate):
        def collision_check_using_contact_graspnet(pred_grasps):
            collision_free_grasps = []
            for grasps in pred_grasps:
                self.action.scene_mngr.set_gripper_pose(grasps)
                if not self.action._collide(is_only_gripper=True):
                    collision_free_grasps.append(grasps)

            return np.array(collision_free_grasps)

        pred_grasps_world = {}
        pred_grasps_world_augment = {}
        pred_grasps_cam_augment = {}
        collision_free_grasps = []
        # Z축으로 90도 돌려야함.
        z_90_matrix = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        pred_grasps_world[obj_to_manipulate] = self.w_T_cam @ pred_grasps_cam

        eef_poses = np.zeros((pred_grasps_world[obj_to_manipulate].shape))
        for i in range(len(pred_grasps_world[obj_to_manipulate])):
            eef_poses[i] = self.compute_eef_pose_from_tcp_pose(
                pred_grasps_world[obj_to_manipulate][i], depth=self.gripper_depth[i]
            )

        pred_grasps_world[obj_to_manipulate] = eef_poses

        collision_free_grasps = collision_check_using_contact_graspnet(
            pred_grasps_world[obj_to_manipulate]
        )

        if not len(collision_free_grasps) or self.bench_num == 2:
            pred_grasps_cam_augment[obj_to_manipulate] = pred_grasps_cam @ z_90_matrix
            pred_grasps_world_augment[obj_to_manipulate] = (
                self.w_T_cam @ pred_grasps_cam_augment[obj_to_manipulate]
            )

            collision_free_grasps = collision_check_using_contact_graspnet(
                pred_grasps_world_augment[obj_to_manipulate]
            )

        augmented_grasps = []
        if not len(collision_free_grasps) or self.bench_num == 2:
            pred_grasps_world_augment[obj_to_manipulate] = np.vstack(
                [pred_grasps_world_augment[obj_to_manipulate], pred_grasps_world[obj_to_manipulate]]
            )
            for grasps in pred_grasps_world_augment[obj_to_manipulate]:
                self.action.scene_mngr.set_gripper_pose(grasps)
                tcp_pose = self.action.scene_mngr.scene.robot.gripper.get_gripper_tcp_pose()

                for tcp_pose_ in get_heuristic_eef_pose(tcp_pose):
                    eef_pose_ = (
                        self.action.scene_mngr.scene.robot.gripper.compute_eef_pose_from_tcp_pose(
                            tcp_pose_
                        )
                    )
                    augmented_grasps.append(eef_pose_)

        if augmented_grasps:
            augmented_grasps = np.array(augmented_grasps)
            print("Augment 2 y axis rotation from -pi/3 ~ pi/3 : ", augmented_grasps.shape)
            pred_grasps_world_augment[obj_to_manipulate] = augmented_grasps

            collision_free_grasps = collision_check_using_contact_graspnet(
                pred_grasps_world_augment[obj_to_manipulate]
            )
            print("Collision free grasps step 3 : ", collision_free_grasps.shape)

        return collision_free_grasps

    def compute_eef_pose_from_tcp_pose(self, tcp_pose=np.eye(4), depth=0.01):
        eef_pose = np.eye(4)
        eef_pose[:3, :3] = tcp_pose[:3, :3]
        eef_pose[:3, 3] = tcp_pose[:3, 3] - np.dot(
            abs(self.action.scene_mngr.scene.robot.gripper.tcp_position[-1]) + depth / 2,
            tcp_pose[:3, 2],
        )
        return eef_pose

    def get_grasp(
        self,
        init_scene,
        next_node=None,
        current_node=None,
    ):
        obj_to_manipulate = current_node["action"]["rearr_obj_name"]
        if self.bench_num == 2:
            min_size = 0.6
        else:
            min_size = 0.4

        if next_node != None:
            self.action.get_mixed_scene_on_current(
                next_scene=next_node["state"],
                current_scene=current_node["state"],
                obj_to_manipulate=obj_to_manipulate,
            )
            pc, pc_segments, pc_color, count = get_obj_point_clouds(
                init_scene, self.action.scene_mngr.scene, obj_to_manipulate
            )
        else:
            pc, pc_segments, pc_color, count = get_obj_point_clouds(
                init_scene, current_node["state"], obj_to_manipulate
            )
        table_point_cloud, table_color = get_support_space_point_cloud(
            init_scene, current_node["state"]
        )

        # in pc_utils
        all_pc = np.vstack([pc, table_point_cloud])
        all_color = np.vstack([pc_color, table_color])

        cam_pc, pc_segments = self.get_pc_from_camera_point_of_view(
            all_pc, pc_segments, obj_to_manipulate
        )

        pc_regions, _ = self.extract_3d_cam_boxes(cam_pc[:, :3], pc_segments)

        data = {}
        # data['point_clouds'] = torch.unsqueeze(torch.from_numpy(pc_sampled),0).cuda().to(torch.float32)
        data["point_clouds"] = (
            torch.unsqueeze(torch.from_numpy(pc_regions[obj_to_manipulate]), 0)
            .cuda()
            .to(torch.float32)
        )

        with torch.no_grad():
            end_points = self.net(data)
            grasp_preds = self.pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        print("gg_arr", len(gg_array))
        segments_inds = self.filter_segment(
            gg_array[:, 13:16], pc_segments[obj_to_manipulate], thres=0.001
        )
        print("filtered :", len(segments_inds))
        grasp_to_manipulate = gg_array[segments_inds]
        # grasp_to_manipulate = grasp_to_manipulate[grasp_to_manipulate[:, 0] > 0.3]
        grasp_thresh = grasp_to_manipulate[grasp_to_manipulate[:, 12] > 0]

        gg = self.GraspGroup(grasp_thresh)
        print("thresh", len(gg))
        gripper_tcp_point = self.get_grasp_pose_for_scale_balance_grasp(
            gg.rotation_matrices, gg.translations
        )
        self.gripper_depth = gg.depths

        collision_free_grasps = self.change_grasp_to_world_coord(
            gripper_tcp_point, obj_to_manipulate
        )

        return collision_free_grasps


class Grasp_Using_Contact_GraspNet(Grasp_Using_AI_model):
    def __init__(self, action, robot_name, bench_num=0):
        super().__init__(action, robot_name, bench_num)

        tf.reset_default_graph()
        tf.disable_eager_execution()
        physical_devices = tf.config.list_physical_devices("GPU")

        if physical_devices:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    physical_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
                )  # 4GB 제한
            except RuntimeError as e:
                print(e)

        home_path = os.path.expanduser("~")
        sys.path.append(os.path.join(home_path, "contact_graspnet/contact_graspnet"))

        import config_utils

        # from contact_graspnet import contact_graspnet
        from contact_grasp_estimator import GraspEstimator
        from visualization_utils import visualize_grasps

        self.visualize_grasps = visualize_grasps
        FLAGS = self.argparse()
        self.global_config = config_utils.load_config(
            FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs
        )

        self.grasp_estimator = GraspEstimator(self.global_config)
        self.grasp_estimator.build_network(robot_name)

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        checkpoint_dir = (
            home_path + "/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001/"
        )
        self.saver.restore(
            self.sess,
            home_path
            + "/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001/model.ckpt-144144",
        )
        self.bench_num = bench_num
        if bench_num < 2:
            self.T_cam = np.array(
                [
                    [6.12323400e-17, -8.66025404e-01, 5.00000000e-01, 1.60000008e-01],
                    [1.00000000e00, 5.30287619e-17, -3.06161700e-17, 6.34369494e-01],
                    [-0.00000000e00, 5.00000000e-01, 8.66025404e-01, 1.43132538e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            )

        if bench_num == 2:
            self.T_cam = np.array(
                [
                    [1.0, 0.0, 0.0, 1.46763353],
                    [0.0, -0.09983342, -0.99500417, -0.42326634],
                    [-0.0, 0.99500417, -0.09983342, 0.83645545],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        if bench_num == 3:
            self.T_cam = np.array(
                [
                    [6.12323400e-17, 8.66025404e-01, -5.00000000e-01, -8.39999992e-01],
                    [-1.00000000e00, 5.30287619e-17, -3.06161700e-17, 8.34369494e-01],
                    [-0.00000000e00, 5.00000000e-01, 8.66025404e-01, 1.43132538e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            )

        try:
            w_T_cam = self.action.scene_mngr.scene.objs["table"].h_mat @ self.T_cam
        except:
            w_T_cam = self.action.scene_mngr.scene.objs["shelves"].h_mat @ self.T_cam
        m_ = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.w_T_cam = w_T_cam.dot(m_)

    def argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--ckpt_dir",
            default="checkpoints/scene_test_2048_bs3_hor_sigma_001",
            help="Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]",
        )
        parser.add_argument(
            "--np_path",
            default="test_data/7.npy",
            help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"',
        )
        parser.add_argument("--png_path", default="", help="Input data: depth map png in meters")
        parser.add_argument(
            "--K",
            default=None,
            help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"',
        )
        parser.add_argument(
            "--z_range", default=[0.2, 1.8], help="Z value threshold to crop the input point cloud"
        )
        parser.add_argument(
            "--local_regions",
            action="store_true",
            default=False,
            help="Crop 3D local regions around given segments.",
        )
        parser.add_argument(
            "--filter_grasps",
            action="store_true",
            default=False,
            help="Filter grasp contacts according to segmap.",
        )
        parser.add_argument(
            "--skip_border_objects",
            action="store_true",
            default=False,
            help="When extracting local_regions, ignore segments at depth map boundary.",
        )
        parser.add_argument(
            "--forward_passes",
            type=int,
            default=1,
            help="Run multiple parallel forward passes to mesh_utils more potential contact points.",
        )
        parser.add_argument(
            "--segmap_id", type=int, default=0, help="Only return grasps of the given object id"
        )
        parser.add_argument(
            "--arg_configs", nargs="*", type=str, default=[], help="overwrite config parameters"
        )

        FLAGS = parser.parse_args(args=[])
        return FLAGS

    def get_region_to_manipulate(self, pc_full, pc_segments, min_size, obj_name):
        pc_regions, _ = self.grasp_estimator.extract_3d_cam_boxes(pc_full, pc_segments, min_size)
        return pc_regions[obj_name]

    def generate_grasp(self, pc_region, pc_segment, obj_name):
        """
        pc_region : pc_cube [N,3] at camera view
        pc_segments : pc that want to manipulate at camera view
        obj_name : pc want to manipulate
        """
        pred_grasps_cam, scores, contact_pts, gripper_openings = {}, {}, {}, {}
        forward_passes = 1
        local_regions = True

        (
            pred_grasps_cam[obj_name],
            scores[obj_name],
            contact_pts[obj_name],
            gripper_openings[obj_name],
        ) = self.grasp_estimator.predict_grasps(
            self.sess, pc_region, convert_cam_coords=True, forward_passes=forward_passes
        )

        # Filter grasps
        segment_keys = contact_pts.keys()
        for k in segment_keys:
            if np.any(pc_segment) and np.any(contact_pts[k]):
                segment_idcs = self.grasp_estimator.filter_segment(
                    contact_pts[k], pc_segment, thres=0.005
                )

                pred_grasps_cam[k] = pred_grasps_cam[k][segment_idcs]
                scores[k] = scores[k][segment_idcs]
                contact_pts[k] = contact_pts[k][segment_idcs]
                try:
                    gripper_openings[k] = gripper_openings[k][segment_idcs]
                except:
                    print("skipped gripper openings {}".format(gripper_openings[k]))
                if local_regions and np.any(pred_grasps_cam[k]):
                    print("Generated {} grasps for object {}".format(len(pred_grasps_cam[k]), k))
            else:
                print(
                    "skipping obj {} since  np.any(pc_segments[k]) {} and np.any(contact_pts[j]) is {}".format(
                        k, np.any(pc_segment), np.any(contact_pts[k])
                    )
                )

        return pred_grasps_cam, scores, contact_pts, gripper_openings

    def visualization(self, pc, pred_grasps, scores, pc_colors):
        """
        pc : The point cloud of the entire scene
        """
        self.visualize_grasps(pc, pred_grasps, scores, plot_opencv_cam=True, pc_colors=pc_colors)

    def get_pc_from_camera_point_of_view(
        self,
        all_pc,
        pc_segments,
        obj_to_manipulate,
    ):
        cam_T_w = get_inverse_homogeneous(self.w_T_cam)
        ones_arr = np.full((len(all_pc), 1), 1)
        w_pc = np.hstack((all_pc, ones_arr))

        cam_pc = np.dot(cam_T_w, w_pc.T).T

        # next_pc_segment도 변경해줘
        ones_arr = np.full((len(pc_segments[obj_to_manipulate]), 1), 1)
        w_pc = np.hstack((pc_segments[obj_to_manipulate], ones_arr))

        pc_segments[obj_to_manipulate] = np.dot(cam_T_w, w_pc.T).T

        return cam_pc, pc_segments

    def change_grasp_to_world_coord(self, pred_grasps_cam, obj_to_manipulate):
        def collision_check_using_contact_graspnet(pred_grasps):
            collision_free_grasps = []
            for grasps in pred_grasps:
                self.action.scene_mngr.set_gripper_pose(grasps)
                if not self.action._collide(is_only_gripper=True):
                    collision_free_grasps.append(grasps)

            return np.array(collision_free_grasps)

        pred_grasps_world = {}
        pred_grasps_world_augment = {}
        pred_grasps_cam_augment = {}
        collision_free_grasps = []
        # Z축으로 90도 돌려야함.
        z_90_matrix = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        pred_grasps_cam_augment[obj_to_manipulate] = (
            pred_grasps_cam[obj_to_manipulate] @ z_90_matrix
        )
        pred_grasps_world[obj_to_manipulate] = (
            self.w_T_cam @ pred_grasps_cam_augment[obj_to_manipulate]
        )
        # print("Generated Grasp in world coord :", pred_grasps_world[obj_to_manipulate].shape)

        collision_free_grasps = collision_check_using_contact_graspnet(
            pred_grasps_world[obj_to_manipulate]
        )
        # print("Collision free grasps step 1 : ", collision_free_grasps.shape)

        if not len(collision_free_grasps) or self.bench_num == 2:
            pred_grasps_world_augment[obj_to_manipulate] = (
                self.w_T_cam @ pred_grasps_cam[obj_to_manipulate]
            )

            collision_free_grasps = collision_check_using_contact_graspnet(
                pred_grasps_world_augment[obj_to_manipulate]
            )
            # print("Collision free grasps step 2 : ", collision_free_grasps.shape)

        augmented_grasps = []
        if not len(collision_free_grasps) or self.bench_num == 2:
            pred_grasps_world_augment[obj_to_manipulate] = np.vstack(
                [pred_grasps_world_augment[obj_to_manipulate], pred_grasps_world[obj_to_manipulate]]
            )
            for grasps in pred_grasps_world_augment[obj_to_manipulate]:
                self.action.scene_mngr.set_gripper_pose(grasps)
                tcp_pose = self.action.scene_mngr.scene.robot.gripper.get_gripper_tcp_pose()

                for tcp_pose_ in get_heuristic_eef_pose(tcp_pose):
                    eef_pose_ = (
                        self.action.scene_mngr.scene.robot.gripper.compute_eef_pose_from_tcp_pose(
                            tcp_pose_
                        )
                    )
                    # self.action.scene_mngr.set_gripper_pose(eef_pose_)
                    augmented_grasps.append(eef_pose_)

        if augmented_grasps:
            augmented_grasps = np.array(augmented_grasps)
            print("Augment 2 y axis rotation from -pi/3 ~ pi/3 : ", augmented_grasps.shape)
            pred_grasps_world_augment[obj_to_manipulate] = augmented_grasps

            collision_free_grasps = collision_check_using_contact_graspnet(
                pred_grasps_world_augment[obj_to_manipulate]
            )
            print("Collision free grasps step 3 : ", collision_free_grasps.shape)

        return collision_free_grasps

    def remove_mixed_scene(self):
        self.action.remove_mixed_scene()

    def get_grasp(
        self,
        init_scene,
        next_node=None,
        current_node=None,
    ):
        obj_to_manipulate = current_node["action"]["rearr_obj_name"]

        if self.bench_num == 2:
            min_size = 0.6
        else:
            min_size = 0.4
        if next_node != None:
            self.action.get_mixed_scene_on_current(
                next_scene=next_node["state"],
                current_scene=current_node["state"],
                obj_to_manipulate=obj_to_manipulate,
            )
            pc, pc_segments, pc_color, count = get_obj_point_clouds(
                init_scene, self.action.scene_mngr.scene, obj_to_manipulate
            )
        else:
            pc, pc_segments, pc_color, count = get_obj_point_clouds(
                init_scene, current_node["state"], obj_to_manipulate
            )
        table_point_cloud, table_color = get_support_space_point_cloud(
            init_scene, current_node["state"]
        )

        # in pc_utils
        all_pc = np.vstack([pc, table_point_cloud])
        all_color = np.vstack([pc_color, table_color])

        cam_pc, pc_segments = self.get_pc_from_camera_point_of_view(
            all_pc, pc_segments, obj_to_manipulate
        )

        pc_region_ = self.get_region_to_manipulate(
            cam_pc[:, :3], pc_segments, min_size=min_size, obj_name=obj_to_manipulate
        )

        pred_grasps_cam, s_, c_, g_o_ = {}, {}, {}, {}

        pred_grasps_cam, s_, c_, g_o_ = self.generate_grasp(
            pc_region_, pc_segments[obj_to_manipulate], obj_to_manipulate
        )
        collision_free_grasps = self.change_grasp_to_world_coord(pred_grasps_cam, obj_to_manipulate)

        # self.action.remove_mixed_scene()

        return collision_free_grasps

    # def get_all_grasps(self, grasp_set):
    #     grasp_poses_not_collision = self.action.get_all_grasp_poses_not_collision(grasp_set)
    #     return grasp_poses_not_collision
