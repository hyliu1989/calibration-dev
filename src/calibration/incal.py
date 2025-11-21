"""A module for camera intrinsic calibration with charuco board images."""
from typing import Sequence

import cv2 as cv
import numpy as np
import numpy.typing as npt


def process_image_with_charuco_board(
    images: list[npt.NDArray],
    board: cv.aruco.CharucoBoard,
    detector_parameters: cv.aruco.DetectorParameters | None = None,
) -> tuple[list[npt.NDArray], list[npt.NDArray]]:
    """Detects the charuco board point coordinates and their corresponding image points.

    Returns:
        A tuple of the following:
        - a list of sublists of detected points in 3D in the charuco board coordinate system, one sublist for one input
            image.
        - a list of sublists of corresponding  points in 2D in the image coordinate system, one sublist for one input
            image.
    """
    dictionary = board.getDictionary()
    if detector_parameters is None:
        detector_parameters = cv.aruco.DetectorParameters()

    img_shape = images[0].shape
    object_point_sets = []
    image_point_sets = []
    for i, img in enumerate(images):
        assert img.shape == img_shape

        try:
            marker_corners, marker_ids, rejected_pts = cv.aruco.detectMarkers(
                img, dictionary, parameters=detector_parameters
            )
            charuco_retval, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, img, board
            )
            if charuco_retval == 0:
                print(f"images[{i}] does not yield a valid calibration pattern!!")
                continue
            print("# of retrieved points:", charuco_retval)
    
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            # obj_points is in (x, y, z) of 3D points of the board.
            # img_points is in (x, y) of 2D points in the projected image.
        except Exception as e:
            print(f"Error happens when processing images[{i}]!!")
            print(e)
            continue

        if len(obj_points) < 4:
            print(f"Not enough detected points in images[{i}]")
            continue
        
        object_point_sets.append(obj_points)
        image_point_sets.append(img_points)

        for obj_points, img_points in zip(obj_points, img_points):
            assert len(obj_points) == len(img_points)

    return object_point_sets, image_point_sets


def filter_points_at_fov_edge(
    image_points, k_mat, fisheye=True, fov_limit_degree=80.0, object_points=None
) -> list[npt.NDArray] | tuple[list[npt.NDArray], list[npt.NDArray]]:
    assert fisheye is True
    fx = k_mat[0, 0]
    fy = k_mat[1, 1]
    cx = k_mat[0, 2]
    cy = k_mat[1, 2]
    fov_limit = np.deg2rad(fov_limit_degree)
    uv = np.squeeze(image_points) - np.array([cx, cy])
    uv = uv / np.array([fx, fy])
    theta = np.linalg.norm(uv, axis=-1)
    mask_kept = theta < fov_limit
    if object_points is not None:
        return image_points[mask_kept], object_points[mask_kept]
    return image_points[mask_kept]


def display_intrinsics_calibration_result(
    res: tuple[float, npt.NDArray, npt.NDArray, Sequence[npt.NDArray], Sequence[npt.NDArray]],
) -> None:
    print("- loss:", res[0])
    print("- K matrix:")
    print(res[1])
    print("- distortion:", res[2])


def process_v0(
    object_point_sets: list[npt.NDArray],
    image_point_sets: list[npt.NDArray],
    verbose: bool = False,
):
    """Calibration process version 0 for a certain project.

    The first of each list, object_point_set and image_point_set, is expected to be derived from an image where the
    checkerboard is at the center of the image.
    """
    # Stage 1
    k_mat_init = np.array(
        [
            [666, 0, 1928,],
            [0, 666, 1460],
            [0, 0, 1],
        ],
        dtype=np.float32
    )
    distortion_init = np.zeros(4, dtype=np.float32)
    flags = (
        cv.fisheye.CALIB_FIX_SKEW
        | cv.fisheye.CALIB_USE_INTRINSIC_GUESS
        | cv.fisheye.CALIB_FIX_K1
        | cv.fisheye.CALIB_FIX_K2
        | cv.fisheye.CALIB_FIX_K3
        | cv.fisheye.CALIB_FIX_K4
        | cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    )
    calibration_result0 = cv.fisheye.calibrate(
        object_point_sets[0: 1],
        image_point_sets[0: 1],
        (0, 0,),
        K=k_mat_init,
        D=distortion_init,
        flags=flags,
    )
    if verbose:
        display_intrinsics_calibration_result(calibration_result0)

    # stage 2
    k_mat = calibration_result0[1]
    distortion_init = np.zeros(4, dtype=np.float32)
    image_restricted_point_sets = []
    object_restricted_point_sets = []
    for ips, ops in zip(image_point_sets, object_point_sets):
        filtered = filter_points_at_fov_edge(ips, k_mat, object_points=ops, fov_limit_degree=70.0)
        image_restricted_point_sets.append(filtered[0])
        object_restricted_point_sets.append(filtered[1])
    flags = (
        cv.fisheye.CALIB_FIX_SKEW
        | cv.fisheye.CALIB_USE_INTRINSIC_GUESS
        | cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    )
    calibration_result1 = cv.fisheye.calibrate(
        object_restricted_point_sets,
        image_restricted_point_sets,
        (0, 0),
        K=k_mat.copy(),
        D=distortion_init,
        flags=flags,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 500, 1e-3),
    )
    if verbose:
        display_intrinsics_calibration_result(calibration_result1)

    return calibration_result1


def save_intrinsics(filepath, res, id_cam) -> None:
    K = res[1]
    D = res[2]
    assert id_cam in [1, 2]
    assert K[2, 2] == 1.0
    content = f"""i{id_cam}_model = 1
i{id_cam}_fx = {K[0, 0]}
i{id_cam}_fy = {K[1, 1]}
i{id_cam}_cx = {K[0, 2]}
i{id_cam}_cy = {K[1, 2]}
i{id_cam}_k1 = {D[0]}
i{id_cam}_k2 = {D[1]}
i{id_cam}_k3 = {D[2]}
i{id_cam}_k4 = {D[3]}
i{id_cam}_k5 = 0.0
i{id_cam}_k6 = 0.0
i{id_cam}_p1 = 0.0
i{id_cam}_p2 = 0.0
"""
    with open(filepath, "w") as f:
        f.write(content)
    return
