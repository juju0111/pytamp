import os
import sys
import argparse
import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from pykin.utils.transform_utils import get_inverse_homogeneous


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
        from contact_graspnet import contact_graspnet
        from contact_grasp_estimator import GraspEstimator
        from visualization_utils import visualize_grasps

        self.visualize_grasps = visualize_grasps
        self.action = action
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
            T_cam = np.array(
                [
                    [6.12323400e-17, -8.66025404e-01, 5.00000000e-01, 1.60000008e-01],
                    [1.00000000e00, 5.30287619e-17, -3.06161700e-17, 6.34369494e-01],
                    [-0.00000000e00, 5.00000000e-01, 8.66025404e-01, 1.63132538e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            )

        assert T_cam.shape == (4, 4), "cam_transformation shape must be (4,4)"

        w_T_cam = self.action.scene_mngr.scene.objs["table"].h_mat @ T_cam
        m_ = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        w_T_cam = w_T_cam.dot(m_)

        cam_T_w = get_inverse_homogeneous(w_T_cam)
        ones_arr = np.full((len(all_pc), 1), 1)
        w_pc = np.hstack((all_pc, ones_arr))

        cam_pc = np.dot(cam_T_w, w_pc.T).T

        # next_pc_segment도 변경해줘
        ones_arr = np.full((len(pc_segments[obj_to_manipulate]), 1), 1)
        w_pc = np.hstack((pc_segments[obj_to_manipulate], ones_arr))

        pc_segments[obj_to_manipulate] = np.dot(cam_T_w, w_pc.T).T

        return cam_pc, pc_segments
