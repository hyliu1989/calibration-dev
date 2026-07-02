"""Utilities for calibration angles."""
import numpy.typing as npt
from pyhammer.trinsics import BaselineFrameStereoState
from scipy.spatial.transform import Rotation as _Rotation


def baseline_frame_stereo_state_from_two_rotations(
    rot_mat_1: npt.NDArray,
    rot_mat_2: npt.NDArray,
    baseline_meters: float,
) -> BaselineFrameStereoState:
    """Converts from two rotation matrices and baseline to BaselineFrameStereoState.

    Args:
        rot_mat_1: Rotation of the 1st camera. This matrix rotates the axes of the rectified camera
            to the raw camera.
        rot_mat_2: Rotation of the 2nd camera. This matrix rotates the axes of the rectified camera
            to the raw camera.
        baseline_meters: Baseline length in meters.
    """
    angles1 = _Rotation.from_matrix(rot_mat_1).as_euler("zyx", degrees=True)
    angles2 = _Rotation.from_matrix(rot_mat_2).as_euler("zyx", degrees=True)
    return BaselineFrameStereoState(
        angles1[2],  # euler x
        angles1[1],  # euler y
        angles1[0],  # euler z
        angles2[2],  # euler x
        angles2[1],  # euler y
        angles2[0],  # euler z
        t_norm=baseline_meters,
    )
