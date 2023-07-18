import os
import sys
import argparse
import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from pykin.utils.transform_utils import get_inverse_homogeneous
from pytamp.utils.heuristic_utils import get_heuristic_eef_pose
from pytamp.utils.point_cloud_utils import get_mixed_scene
from pytamp.utils.point_cloud_utils import get_obj_point_clouds
from pytamp.utils.point_cloud_utils import get_support_space_point_cloud


class Grasp_Using_Contact_GraspNet:
    def __init__(self, action):
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
        self.action = action
        self.T_cam = np.eye(4)
        self.w_T_cam = np.eye(4)
        FLAGS = self.argparse()
        self.global_config = config_utils.load_config(
            FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs
        )

        self.grasp_estimator = GraspEstimator(self.global_config)
        self.grasp_estimator.build_network()

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
                        k, np.any(pc_segment), np.any(contact_pts[j])
                    )
                )

        return pred_grasps_cam, scores, contact_pts, gripper_openings

    def visualization(self, pc, pred_grasps, scores, pc_colors):
        """
        pc : The point cloud of the entire scene
        """
        self.visualize_grasps(pc, pred_grasps, scores, plot_opencv_cam=True, pc_colors=pc_colors)

    def get_pc_from_camera_point_of_view(self, all_pc, pc_segments, obj_to_manipulate, T_cam=None):
        if T_cam == None:
            self.T_cam = np.array(
                [
                    [6.12323400e-17, -8.66025404e-01, 5.00000000e-01, 1.60000008e-01],
                    [1.00000000e00, 5.30287619e-17, -3.06161700e-17, 6.34369494e-01],
                    [-0.00000000e00, 5.00000000e-01, 8.66025404e-01, 1.63132538e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            )
        else:
            self.T_cam = T_cam
        assert self.T_cam.shape == (4, 4), "cam_transformation shape must be (4,4)"

        w_T_cam = self.action.scene_mngr.scene.objs["table"].h_mat @ self.T_cam
        m_ = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.w_T_cam = w_T_cam.dot(m_)

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
        print("Generated Grasp in world coord :", pred_grasps_world[obj_to_manipulate].shape)

        collision_free_grasps = collision_check_using_contact_graspnet(
            pred_grasps_world[obj_to_manipulate]
        )
        print("Collision free grasps step 1 : ", collision_free_grasps.shape)

        if not len(collision_free_grasps):
            pred_grasps_world_augment[obj_to_manipulate] = (
                self.w_T_cam @ pred_grasps_cam[obj_to_manipulate]
            )
            print(
                "Augment 1 _z axis 90' rotation ",
                pred_grasps_world_augment[obj_to_manipulate].shape,
                pred_grasps_world[obj_to_manipulate].shape,
            )

            collision_free_grasps = collision_check_using_contact_graspnet(
                pred_grasps_world_augment[obj_to_manipulate]
            )
            print("Collision free grasps step 2 : ", collision_free_grasps.shape)

        augmented_grasps = []
        if not len(collision_free_grasps):
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
                    self.action.scene_mngr.set_gripper_pose(eef_pose_)
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
            cam_pc[:, :3], pc_segments, min_size=0.4, obj_name=obj_to_manipulate
        )

        pred_grasps_cam, s_, c_, g_o_ = {}, {}, {}, {}

        pred_grasps_cam, s_, c_, g_o_ = self.generate_grasp(
            pc_region_, pc_segments[obj_to_manipulate], obj_to_manipulate
        )
        collision_free_grasps = self.change_grasp_to_world_coord(pred_grasps_cam, obj_to_manipulate)

        # self.action.remove_mixed_scene()

        return collision_free_grasps

    def get_all_grasps(self, collision_free_grasps):
        grasp_poses = list(self.action.get_all_grasps_from_grasps(collision_free_grasps))
        grasp_poses_not_collision = list(self.action.get_all_grasp_poses_not_collision(grasp_poses))
        self.action.remove_mixed_scene()
        return grasp_poses_not_collision
