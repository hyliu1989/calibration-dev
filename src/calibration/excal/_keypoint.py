import logging

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt
from pyhammer.trinsics import BaselineFrameStereoState
from pyhammer.trinsics import baseline_frame_stereo_state_from_two_rotations


logger = logging.getLogger(__name__)
__all__ = ["recover_pose"]


def recover_pose(
    points_sets: list[tuple[npt.NDArray, npt.NDArray]],
    undistorted_k: npt.NDArray,
    undistorted_into: str = "rectilinear",
    undistort_rotation: npt.NDArray | None = None,
    verbose: bool = False,
    debug: bool = True,
    debug_name: str = "debug_recover_pose"
) -> BaselineFrameStereoState:
    """Recovers the relative pose between two images given matched points.

    Args:
        points_sets: A list of tuples, each containing two numpy arrays of shape (num_points, 3) representing
            matched points in the two images in (x, y, confidence) format.
        undistorted_k: The undistorted camera intrinsic matrix (3x3 numpy array).
        undistorted_into: A string indicating the intrinsics model of the undistorted image. Valid values are
            "rectilinear", "cylindrical_x", "cylindrical_y".
        undistort_rotation: An optional rotation matrix (3x3 numpy array) applied during undistortion. The rotation
            matrix transforms the coordinates in original camera frame coordinate to the undistorted image coordinate.
        verbose: If True, prints verbose information during processing.
        debug: If True, generates a debug plot showing the essential matrix inner product errors.
        debug_name: The name to use for the debug plot file.
    """
    uv_image0_sets = []
    uv_image1_sets = []
    fx = undistorted_k[0, 0]
    fy = undistorted_k[1, 1]
    cx = undistorted_k[0, 2]
    cy = undistorted_k[1, 2]

    if undistorted_into == "rectilinear":
        undistorted_k_inv = np.linalg.inv(undistorted_k)
        for xyc0s, xyc1s in points_sets:
            # xyc0s, xyc1s:(num_points, 3) where 3 is for (x, y, confidence)
            # xy1_image0 and xy1_image1 will be of shape (3, num_points) where 3 is for (x, y, 1)
            xy1_image0 = xyc0s.T.copy()
            xy1_image0[2] = 1.0
            uv_image0 = undistorted_k_inv[:2, :] @ xy1_image0
            uv_image0_sets.append(uv_image0)

            xy1_image1 = xyc1s.T.copy()
            xy1_image1[2] = 1.0
            uv_image1 = undistorted_k_inv[:2, :] @ xy1_image1
            uv_image1_sets.append(uv_image1)
    elif undistorted_into == "cylindrical_x":
        delta_azimuth_x = 1 / undistorted_k[0, 0]  # in radians
        for xyc0s, xyc1s in points_sets:
            # xyc0s, xyc1s:(num_points, 3) where 3 is for (x, y, confidence)
            azimuth_x = (xyc0s[:, 0] - cx) * delta_azimuth_x
            x = fx * np.sin(azimuth_x)
            y = (fx / fy) * (xyc0s[:, 1] - cy)
            z = fx * np.cos(azimuth_x)
            uv_image0_sets.append(np.vstack((x, y, z)))

            azimuth_x = xyc1s[:, 0] * delta_azimuth_x
            x = fx * np.sin(azimuth_x)
            y = (fx / fy) * xyc1s[:, 1]
            z = fx * np.cos(azimuth_x)
            uv_image1_sets.append(np.vstack((x, y, z)))
    elif undistorted_into == "cylindrical_y":
        delta_azimuth_y = 1 / undistorted_k[0, 0]  # in radians
        for xyc0s, xyc1s in points_sets:
            # xyc0s, xyc1s:(num_points, 3) where 3 is for (x, y, confidence)
            azimuth_y = (xyc0s[:, 1] - cy) * delta_azimuth_y
            x = (fy / fx) * (xyc0s[:, 0] - cx)
            y = fy * np.sin(azimuth_y)
            z = fy * np.cos(azimuth_y)
            uv_image0_sets.append(np.vstack((x, y, z)))

            azimuth_y = xyc1s[:, 1] * delta_azimuth_y
            x = (fy / fx) * xyc1s[:, 0]
            y = fy * np.sin(azimuth_y)
            z = fy * np.cos(azimuth_y)
            uv_image1_sets.append(np.vstack((x, y, z)))
    else:
        raise ValueError(f"Invalid undistorted_into value: {undistorted_into}")
    uv_image0 = np.hstack(uv_image0_sets).T
    uv_image1 = np.hstack(uv_image1_sets).T

    f_original = undistorted_k[0, 0]
    f = 1.0
    e_mat, mask = cv.findEssentialMat(uv_image0, uv_image1, focal=f, pp=(0.0, 0.0), threshold=1 / f_original)
    if verbose:
        logger.info("The percentage of the point pairs used:", 100 * mask.sum() / mask.size, "%")

    ret_val, rot_mat, t_vec_proper, mask2 = cv.recoverPose(
        E=e_mat, points1=uv_image0, points2=uv_image1, focal=1.0, pp=(0.0, 0.0), mask=mask
    )
    state = BaselineFrameStereoState.from_opencv_r_t(rot_mat, t_vec_proper.ravel())
    if verbose:
        logger.info("Horizontal bar R, T:", rot_mat, t_vec_proper.ravel())

    if undistort_rotation is not None:
        # undistort_rotation transforms the coordinates in original camera frame coordinate to the undistorted image
        # coordinate.
        def undo_undistort_rotation(s: BaselineFrameStereoState):
            """Updates the extrinsics that we obtained without considering the undistort rotation."""
            r1 = s.rot_mat_1  # passive rotation, from the undistort image coordinate to the rectified.
            r2 = s.rot_mat_2  # passive rotation, from the undistort image coordinate to the rectified.
            # To make the r1 and r2 transforming original camera frame coordinates to the rectified, we need to apply
            # the rotation from original camera frame to undistorted image frame from the right, making the following:
            #     [R]_{rect <- undist} * [R]_{undist <- original}
            r1 = r1 @ undistort_rotation
            r2 = r2 @ undistort_rotation
            return baseline_frame_stereo_state_from_two_rotations(r1, r2, s.t_norm)
        state = undo_undistort_rotation(state)
    state.global_pitch = 0.0  # reset global pitch to zero since we do not estimate it here.

    if debug:
        def pad_all_one_row(arr):
            all_one_row = np.ones(arr.shape[1], dtype=arr.dtype)
            ret = np.vstack((arr, all_one_row))
            assert ret.shape[0] == 3
            return ret

        def essential_inner_product(ess, pts_image0, pts_image1):
            assert pts_image0.shape[0] == pts_image1.shape[0] == 2
            assert pts_image0.ndim == pts_image1.ndim == 2
            pts_image0 = pad_all_one_row(pts_image0)
            pts_image1 = pad_all_one_row(pts_image1)

            inner_product = [
                (pt_img1[:, np.newaxis].T @ ess @ pt_img0[:, np.newaxis]).item()
                for pt_img0, pt_img1 in zip(pts_image0.T, pts_image1.T)
            ]
            return np.array(inner_product)

        error_to_show = abs(essential_inner_product(e_mat, uv_image0.T, uv_image1.T))

        # Plot the error from each chunk of point matches.
        fh = plt.figure()
        ah = fh.add_subplot(111)
        x_start = 0
        idx_any_image = 0  # any of the image0 or image1
        for chunk_size in [s[idx_any_image].shape[0] for s in points_sets]:
            ah.plot(np.arange(x_start, x_start + chunk_size), error_to_show[x_start:x_start + chunk_size])
            x_start += chunk_size
        percent_essential = 100 * mask.sum() / mask.size
        percent_recover_pose = 100 * mask2.sum() / mask2.size
        ah.set_title(f"{percent_essential:.4f}% pairs used in essential matrix, {percent_recover_pose:.4f}% in recover pose")
        ah.set_xlabel("pair index")
        ah.set_ylabel("E matrix inner product error")
        fh.savefig(f"{debug_name}.png")

    return state
