"""A module performing extrinsics calibration."""
import itertools
import logging
import math
import pickle
from collections.abc import Sequence, Callable
from pathlib import Path

import contexttimer
import cv2 as cv
import numpy as np
import numpy.typing as npt
import pyhammer
import tqdm
from matplotlib import pyplot as plt
from pyhammer.cpyhammer import FomCalculator
from pyhammer.rectification import windowed_planner_wrap
from pyhammer.trinsics import BaselineFrameStereoState
from pyhammer.trinsics import IntrinsicsBase
from scipy.spatial.transform import Rotation

import calibration.optimiz as optimiz
from calibration.specs import PixelDeviationToAngle

logger = logging.getLogger(__name__)


class StateComposer:
    """A class that handles state composition from preconditioned rotations.

    The additional rotation is multiplied from the left of the preconditioned rotation matrices that perform coordinate
    transformation from original cameras to the rectified camera. Hence, the rotation can be thought as rotating in the
    rectified camera coordinate system and the key stone effect in the rectified images will be close to small-angle
    rectification cases.

    Args:
        rot1: The rotation matrix of the first camera. This matrix rotates the axes of the rectified camera to the
            original camera. Equivalently, this matrix performs a coordinate transformation from the original camera to
            the rectified camera.
        rot2: The rotation matrix of the second camera. This matrix rotates the axes of the rectified camera to the
            original camera. Equivalently, this matrix performs a coordinate transformation from the original camera to
            the rectified camera.
        baseline: The baseline length in meters.
    """
    def __init__(self, rot1: npt.NDArray, rot2: npt.NDArray, baseline=1.0):
        self.rot1 = rot1
        self.rot2 = rot2
        self.baseline = baseline

    @classmethod
    def from_state(cls, state: BaselineFrameStereoState) -> "StateComposer":
        return cls(state.rot_mat_1, state.rot_mat_2, baseline=state.t_norm)

    @property
    def initial_state(self) -> BaselineFrameStereoState:
        return pyhammer.trinsics.baseline_frame_stereo_state_from_two_rotations(self.rot1, self.rot2, self.baseline)

    def compose_euler_differential(
        self, axis: str, differential_euler_deg: float, transform: bool = False
    ) -> "BaselineFrameStereoState | StateComposer":
        """Composes a state that is a result of differential rotation to both cameras."""
        assert axis in ("x", "y", "z")
        half_rot_mat = Rotation.from_euler(axis, differential_euler_deg / 2, degrees=True).as_matrix()
        new_rot1 = half_rot_mat @ self.rot1
        new_rot2 = half_rot_mat.T @ self.rot2
        if transform:
            return StateComposer(new_rot1, new_rot2, self.baseline)
        return pyhammer.trinsics.baseline_frame_stereo_state_from_two_rotations(new_rot1, new_rot2, self.baseline)

    def compose_euler_common(
        self, axis: str, common_euler_deg: float, transform: bool = False
    ) -> "BaselineFrameStereoState | StateComposer":
        """Composes a state that is a result of common rotation to both cameras.

        If the axis is "x", then the result should be the same as setting a new global_pitch on top of base one.
        That is,
        compose_euler_common("x", val) is equivalent to
        s = compose_euler_differential("x", 0)
        s.global_pitch += val
        """
        assert axis in ("x", "y", "z")
        rot_mat = Rotation.from_euler(axis, common_euler_deg, degrees=True).as_matrix()
        new_rot1 = rot_mat @ self.rot1
        new_rot2 = rot_mat @ self.rot2
        if transform:
            return StateComposer(new_rot1, new_rot2, self.baseline)
        return pyhammer.trinsics.baseline_frame_stereo_state_from_two_rotations(new_rot1, new_rot2, self.baseline)


class XZSearchHelper:
    """A class that helps compute FOM values from euler x and z angles.

    It is responsible for forming the state from the angles and wrapping the basic FOM calculation so that the input is
    the two angles.
    """
    def __init__(self, image1, i1, image2, i2, fom_calc: FomCalculator, base_state: BaselineFrameStereoState):
        self.image1_gpu = pyhammer.gpu_mat(image1)
        self.i1 = i1
        self.image2_gpu = pyhammer.gpu_mat(image2)
        self.i2 = i2
        self.fom_calc = fom_calc
        self.state_composer = StateComposer.from_state(base_state)

    def state_from_angles(self, euler_x_deg: float, euler_z_deg: float) -> BaselineFrameStereoState:
        composer = self.state_composer
        composer = composer.compose_euler_differential("x", euler_x_deg, transform=True)
        state = composer.compose_euler_differential("z", euler_z_deg)
        return state

    def f(self, euler_x_deg: float, euler_z_deg: float) -> float:
        state = self.state_from_angles(euler_x_deg, euler_z_deg)
        fom = self.fom_calc.calculate(self.image1_gpu, self.i1, self.image2_gpu, self.i2, state)
        return fom


class GoldenSectionEulerAngleOptimizer:
    """A class that performs 1D Golden section search to optimize an euler angle.

    This is derived from the GoldenSectionOptimizer class in the old codebase.

    Args:
        image1: The first image in the stereo pair.
        i1: The intrinsics of the first camera.
        image2: The second image in the stereo pair.
        i2: The intrinsics of the second camera.
        angle_axis: The axis of the euler angle to optimize. Must be one of "x", "y", or "z".
        common_angle: If True, optimize a common angle for both cameras. If False, optimize a differential angle.
            Note that if common_angle is True, angle_axis cannot be "x", since a common rotation around the x-axis does
            not change the extrinsics.
        fom_calc: The FOM calculator to use.
    """
    def __init__(
        self,
        image1: npt.NDArray[np.uint8] | pyhammer.cpyhammer.cv_GpuMat,
        i1: IntrinsicsBase,
        image2: npt.NDArray[np.uint8],
        i2: IntrinsicsBase,
        angle_axis: str,
        common_angle: bool,
        fom_calc: FomCalculator | list[FomCalculator],
    ):
        # The arguments in the initializer sets up the search landscape.
        self.image1_gpu = pyhammer.gpu_mat(image1)
        self.image2_gpu = pyhammer.gpu_mat(image2)
        self.i1 = i1
        self.i2 = i2
        assert angle_axis in ("x", "y", "z")
        self.angle_axis = angle_axis
        self.common_angle = common_angle
        if common_angle:
            assert angle_axis != "x", "Common rotation around x axis does not change the extrinsics."
        self.fom_calc = [fom_calc] if not isinstance(fom_calc, list) else fom_calc

    def optimize(
        self,
        state: BaselineFrameStereoState | StateComposer,
        search_range_deg: float,
        search_tol_deg: float,
    ) -> tuple[BaselineFrameStereoState, float]:
        """Returns the optimized state and the corresponding FOM value."""
        state_composer = state if isinstance(state, StateComposer) else StateComposer.from_state(state)

        def merit_function(angle_deg):
            if self.common_angle:
                s = state_composer.compose_euler_common(self.angle_axis, angle_deg)
            else:
                s = state_composer.compose_euler_differential(self.angle_axis, angle_deg)
            fom = sum([f.calculate(self.image1_gpu, self.i1, self.image2_gpu, self.i2, s) for f in self.fom_calc])
            fom /= len(self.fom_calc)
            return -fom

        optimizer = optimiz.GoldenSectionSearch(merit_function, tol=search_tol_deg)

        # make the search range large enough to cover at least 0.5 * search_range_deg around initial point.
        new_search_range = search_range_deg * 0.5 / optimizer.G2
        x, f_x = optimizer.search_with_range(new_search_range, init_a=0.0)
        if self.common_angle:
            final_state = state_composer.compose_euler_common(self.angle_axis, x)
        else:
            final_state = state_composer.compose_euler_differential(self.angle_axis, x)
        return final_state, -f_x


class InitialCalibration:
    """A class that performs initial calibration of a stereo pair.

    This initial calibration algorithm is based on Hammerhead
    https://github.com/nodarsensor/hammerhead/blob/d86fac3410277296ccb7724fa8f501b0359c3cb5/initial_calibration/src/process.cpp
    with a modification that uses precondition rotations for the two-camera rectification.

    Args:
        image1: The first image in the stereo pair.
        i1: The intrinsics of the first camera.
        image2: The second image in the stereo pair.
        i2: The intrinsics of the second camera.
        planner: The planner to use for rectification.
        rectified_size: The size of the rectified images. If None, use the input image size.
        roi_directive: A string specifying the ROIs when calculating FOM. Valid values are:
            "horizontal": optimize for horizontal stereo, i.e., left-right stereo.
            "vertical": optimize for vertical stereo, i.e., top-bottom stereo.
            "center": Using a smaller central ROI.
            "large_center": Using a larger central ROI.
        verbose: If True, print verbose information during optimization.
    """
    def __init__(
        self,
        image1: npt.NDArray[np.uint8],
        i1: IntrinsicsBase,
        image2: npt.NDArray[np.uint8],
        i2: IntrinsicsBase,
        planner: pyhammer.cpyhammer.AbstractPlanner,
        rectified_size: tuple[int, int] | None = None,
        roi_directive: str = "horizontal",
        verbose: bool = False,
    ):
        self.verbose = verbose
        self._debug_dir = None
        self.image1_gpu = pyhammer.gpu_mat(image1)
        self.image2_gpu = pyhammer.gpu_mat(image2)
        self.i1 = i1
        self.i2 = i2
        self.planner = planner
        input_size = image1.shape[:2][::-1]
        if rectified_size is None:
            rectified_size = input_size
        self._rectified_size: tuple[int, int] = rectified_size
        dummy_state = BaselineFrameStereoState(*([0.0] * 6), t_norm=1.0)
        temp_plan = planner.plan(i1, i2, input_size, dummy_state, rectified_size)
        self._rectified_focal: float = temp_plan.intrinsic1.fx

        # TODO: border is only used when an ROI is specified!
        self.match_border = pyhammer.cpyhammer.FomRoiBorderDirective()
        # border.border = (9 - 1) // 2  # 9 is the matcher patch size
        self.match_border.strict_padding = True

        fom_calc_kwargs = dict(
            border=self.match_border,
            fom_weight_method=1,
            initial_rectified_size=self._rectified_size,
            matcher_id=1,
        )

        self.fom_calc = FomCalculator.create(planner, **fom_calc_kwargs)

        # configure the ROI for FOM calculation.
        self.roi_directive = roi_directive
        if roi_directive == "horizontal":
            row_roi_factor = 1 / 2
            col_roi_factor = 1 / 3
            self.x_roi_planners = [
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, 0, 0)
            ]
            self.z_roi_planners = [
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, col_roi_factor, 0),
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, -col_roi_factor, 0),
            ]
        elif roi_directive == "vertical":
            row_roi_factor = 1 / 3
            col_roi_factor = 1 / 2
            self.x_roi_planners = [
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, 0, 0),
            ]
            self.z_roi_planners = [
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, 0, row_roi_factor),
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, 0, -row_roi_factor)
            ]
        elif roi_directive == "center":
            row_roi_factor = 1 / 2
            col_roi_factor = 1 / 2
            self.x_roi_planners = [
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, 0, 0),
            ]
            self.z_roi_planners = [
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, 0, 0),
            ]
        elif roi_directive == "large_center":
            row_roi_factor = 3 / 4
            col_roi_factor = 3 / 4
            self.x_roi_planners = [
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, 0, 0),
            ]
            self.z_roi_planners = [
                windowed_planner_wrap(planner, col_roi_factor, row_roi_factor, 0, 0),
            ]
        else:
            raise NotImplementedError(f"roi_directive={roi_directive} is not implemented.")
        self.fom_x_roi = [FomCalculator.create(p, **fom_calc_kwargs) for p in self.x_roi_planners]
        self.fom_z_roi = [FomCalculator.create(p, **fom_calc_kwargs) for p in self.z_roi_planners]

    @property
    def rectified_size(self):
        """Returns read-only rectified image size.

        If you want to change this, please create a new InitialCalibration object and specify it at initializer.
        """
        return self._rectified_size

    @property
    def rectified_focal(self):
        """Returns read-only rectified image focal length in pixels."""
        return self._rectified_focal

    @property
    def debug_dir(self) -> Path | None:
        return self._debug_dir

    @debug_dir.setter
    def debug_dir(self, value: str | Path | None):
        if value is None:
            self._debug_dir = None
            return
        p = Path(value)
        p.mkdir(parents=True, exist_ok=True)
        self._debug_dir = p

    @staticmethod
    def get_search_grid(search_range: float, search_step: float, mid_point: float = 0.0) -> list[float]:
        """Returns a sequence of points to search."""
        quotient = search_range / search_step
        num = math.ceil(search_range / search_step)
        # Avoid a precision issue and also don't increate the search range too much.
        if num - quotient > 0.99:
            # likely there is a floating point accuracy issue
            num -= 1
        # Make the number odd.
        if num % 2 == 0:
            num += 1
        # Integer division
        half_num = num // 2

        return [mid_point + i * search_step for i in range(-half_num, half_num + 1)]

    @staticmethod
    def eval_grid2d(
        func: Callable[[float, float], float], grid_0: Sequence[float], grid_1: Sequence[float], verbose: bool = False
    ) -> list[tuple[float, tuple[float, float]]]:
        """Performs a 2D grid evaluation.

        Args:
            func: The function to evaluate. It should take two float arguments and return a float.
            grid_0: The first grid of values to search.
            grid_1: The second grid of values to search.
            verbose: If True, display a progress bar.

        Returns:
            A list of tuples, each containing the function value and the corresponding (x, y) pair.
        """
        enumeration = itertools.product(grid_0, grid_1)
        if verbose:
            total = len(grid_0) * len(grid_1)
            enumeration = tqdm.tqdm(enumeration, total=total)
        ret = []
        for x, y in enumeration:
            val = func(x, y)
            ret.append((val, (x, y)))
        return ret

    @staticmethod
    def double_triangle_area(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Returns twice the signed area of triangle (p0, p1, p2).

        abs(value) / 2 is the actual area.
        """
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        a = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        return a

    def _search_in_euler_xz_space(
        self,
        base_state: BaselineFrameStereoState,
        search_range_x_deg: float,
        search_range_z_deg: float,
        search_step_x_deg: float,
        search_step_z_deg: float,
        total_search_disparity=256,
    ) -> BaselineFrameStereoState:
        """Performs the search of XZ space under a given input base state.

        This is derived from _run_nm_quick_optimize() in the FomCalibration class in the old codebase. The usage is to
        set the euler y angles of the base state and then pass it to this function for a deep XZ search.

        Args:
            base_state: The base state to expand the search space around.
            search_range_x_deg: The full range of Euler_x to search, in degrees. Note that comparing to old terminology,
                search_range_x_deg = 2 * vib_x.
            search_range_z_deg: The full range of Euler_z to search, in degrees. Note that comparing to old terminology,
                search_range_z_deg = 2 * vib_z.
            search_step_x_deg: The step size of Euler_x to search, in degrees. This is equivalent to the old terminology
                delta_x.
            search_step_z_deg: The step size of Euler_z to search, in degrees. This is equivalent to the old terminology
                delta_z.

        Returns:
            The best stereo state found during the search.
        """
        # Set up the function and output handling (update_best_state) for the grid search stage.
        window_spec = dict(window_center_x_offset_ratio=0.0, window_center_y_offset_ratio=0.0)
        fom_calc_kwargs = dict(
            border=self.match_border,
            fom_weight_method=1,
            initial_rectified_size=self._rectified_size,
            matcher_id=2,
        )
        if self.roi_directive == "horizontal":
            window_spec["window_w_ratio"] = 1.0
            window_spec["window_h_ratio"] = 0.5
        elif self.roi_directive == "vertical":
            window_spec["window_w_ratio"] = 0.5
            window_spec["window_h_ratio"] = 1.0
        elif self.roi_directive == "center":
            window_spec["window_w_ratio"] = 0.5
            window_spec["window_h_ratio"] = 0.5
        elif self.roi_directive == "large_center":
            window_spec["window_w_ratio"] = 0.75
            window_spec["window_h_ratio"] = 0.75
        else:
            raise NotImplementedError(f"roi_directive={self.roi_directive} is not implemented.")
        # fom_calc in merit_helper will be set later.
        merit_helper = XZSearchHelper(
            self.image1_gpu, self.i1, self.image2_gpu, self.i2, fom_calc=None, base_state=base_state
        )
        evaluation_list_type = list[tuple[float, tuple[float, float]]]
        grid_search_stage_result = {
            "best_state": (0.0, 0.0),
            "grid_evals": [],  # the grid evaluation list which best_state is from
        }
        def update_best_state(descending_grid_evals: evaluation_list_type, merit_f_helper: XZSearchHelper) -> None:
            best_grid_state: tuple[float, float]
            _, best_grid_state = descending_grid_evals[0]  # descending order
            if best_grid_state == grid_search_stage_result["best_state"]:
                # Always update to the latest given grid if the best states are the same.
                grid_search_stage_result["best_state"] = best_grid_state
                grid_search_stage_result["grid_evals"] = descending_grid_evals
            else:
                # Use the current-scale FoM calculator to evaluate the states to decide which to use as the base.
                fom_from_grid = merit_f_helper.f(*best_grid_state)
                fom_from_prev_best = merit_f_helper.f(*grid_search_stage_result["best_state"])
                if fom_from_grid >= fom_from_prev_best:  # The = sign is important. Update to the new grid when equal.
                    grid_search_stage_result["best_state"] = best_grid_state
                    grid_search_stage_result["grid_evals"] = descending_grid_evals

        # Performs the multiscale grid search.
        # In old codebase:
        #     _grid_search(x0, 4*delta_x, 4*delta_z,      vib_x,      vib_z, 3, self_max_disparity/4)
        #     _grid_search(x0, 2*delta_x, 2*delta_z,  delta_x*4,  delta_z*4, 2, self_max_disparity/4)
        #     _grid_search(x0,   delta_x,   delta_z, delta_x(!), delta_z(!), 1, self_max_disparity/2)  # see note (!)
        args = [
            (3, total_search_disparity // 4, 4 * search_step_x_deg, 4 * search_step_z_deg),
            (2, total_search_disparity // 4, 2 * search_step_x_deg, 2 * search_step_z_deg),
            (1, total_search_disparity // 2, 1 * search_step_x_deg, 1 * search_step_z_deg),
        ]
        x_range = search_range_x_deg
        z_range = search_range_z_deg
        grid_evals: evaluation_list_type = []
        for pyramid, search_disp, x_step, z_step in args:
            if self.verbose:
                logger.info(f"Searching Euler XZ grid for pyramid {pyramid}")
            # Set up the merit function.
            curr_scale_planner = windowed_planner_wrap(self.planner, **window_spec, post_windowing_scale=1 / 2**pyramid)
            merit_helper.fom_calc = FomCalculator.create(
                curr_scale_planner, **fom_calc_kwargs, matcher_num_disparities=search_disp,
            )
            if grid_evals:
                update_best_state(grid_evals, merit_helper)
            # Perform the grid search.
            center = grid_search_stage_result["best_state"]
            x_grid = self.get_search_grid(x_range, x_step, center[0])
            z_grid = self.get_search_grid(z_range, z_step, center[1])
            grid_evals = self.eval_grid2d(merit_helper.f, x_grid, z_grid, verbose=self.verbose)
            grid_evals.sort(key=lambda x: x[0], reverse=True)

            # note(herbert): The ranges in the last iteration differs from the old code. I believe there is a bug in
            #                the old codebase. A `vib` of just 1*delta_x in the old codebase, which has an equivalent
            #                range of 2*delta_x, is too small to cover the segment spanned by a center point and its
            #                two neighbors.
            # This note is referred in codes above. (!)
            x_range = 2 * x_step
            z_range = 2 * z_step

        # Prepare computation object and the initial state for the next stage of the search.
        merit_helper.fom_calc = FomCalculator.create(
            self.planner, **fom_calc_kwargs, matcher_num_disparities=total_search_disparity
        )
        update_best_state(grid_evals, merit_helper)
        del grid_evals

        # Build a simplex for N-M algorithm
        initial_simplex: list[tuple[float, float]] = []
        if merit_helper.f(0.0, 0.0) > merit_helper.f(*grid_search_stage_result["best_state"]):
            # The initial state is already the best.
            if self.verbose:
                logger.info("Create simplex from the initial state instead of the grid search result.")
            initial_simplex.append((0.0, 0.0))
            initial_simplex.append((0.5 * search_step_x_deg, 0.0))
            initial_simplex.append((0.0, 0.5 * search_step_z_deg))
        else:
            # The grid search estimate is better than the initial base state.
            if self.verbose:
                logger.info("Create simplex from the grid search result.")
            simplex_tol = search_step_x_deg * search_step_z_deg / 65
            _, p0 = grid_search_stage_result["grid_evals"][0]
            _, p1 = grid_search_stage_result["grid_evals"][1]
            p2 = None
            # Find the first point that is not collinear with p0 and p1, starting from the points with large FoM.
            for _, point in grid_search_stage_result["grid_evals"][2:]:
                if 0.5 * abs(self.double_triangle_area(p0, p1, point)) > simplex_tol:
                    p2 = point
                    break
            assert p2 is not None
            # Shrink the simplex a bit. (tested and gives better results for precise fine-tuning)
            initial_simplex.append(p0)
            initial_simplex.append((0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])))
            initial_simplex.append((0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1])))

        # Nelder-Mead algorithm
        f_nm = lambda p: -merit_helper.f(p[0], p[1])
        nm_result = optimiz.nelder_mead(f_nm, initial_simplex=initial_simplex, max_iter=2)
        best_angles = nm_result["x"]
        return merit_helper.state_from_angles(*best_angles)

    def calibrate(
        self,
        initial_stereo_state: BaselineFrameStereoState,
        search_range_and_tol_diff_y: tuple[dict[str, float], dict[str, float]] | None = None,
        search_range_and_tol_comm_y: tuple[dict[str, float], dict[str, float]] | None = None,
        search_range_and_tol_diff_x: tuple[dict[str, float], dict[str, float]] | None = None,
        search_range_and_tol_diff_z: tuple[dict[str, float], dict[str, float]] | None = None,
        search_range_and_tol_comm_z: tuple[dict[str, float], dict[str, float]] | None = None,
        search_range_and_tol_comm_y_golden: tuple[dict[str, float], dict[str, float]] | None = None,
        dry_run_for_spec: bool = False,
    ) -> BaselineFrameStereoState | dict[str, tuple[float, float]]:
        """Performs the initial calibration.

        Each dictionary in the search range and tolerance specifications should have at most one of the following keys:
            - "angle": The value of the search range or tolerance in degrees.
            - "pixel": The value of the search range or tolerance in pixels.

        Args:
            initial_stereo_state: The initial stereo state which will be used as a pre-condition.
            search_range_and_tol_diff_y: The search range and tolerance for the differential euler y angle.
            search_range_and_tol_comm_y: The search range and tolerance for the common euler y angle (formerly Tz).
            search_range_and_tol_diff_x: The search range and tolerance for the differential euler x angle.
            search_range_and_tol_diff_z: The search range and tolerance for the differential euler z angle.
            search_range_and_tol_comm_z: The search range and tolerance for the common euler z angle (formerly Ty).
            search_range_and_tol_comm_y_golden: The search range and tolerance for the common euler y angle (formerly
                Tz).
            dry_run_for_spec: If True, do not perform the calibration, just return the parsed search ranges and
                tolerances. This is useful for testing and checking the specification parsing.
        """
        if search_range_and_tol_diff_y is None:
            search_range_and_tol_diff_y = ({"angle": 6.0}, {"pixel": 4.25})
        if search_range_and_tol_comm_y is None:
            search_range_and_tol_comm_y = ({"angle": 0.0}, {"pixel": 4.25})
        if search_range_and_tol_diff_x is None:
            search_range_and_tol_diff_x = ({"angle": 6.0}, {"pixel": 0.5})
        if search_range_and_tol_diff_z is None:
            search_range_and_tol_diff_z = ({"angle": 6.0}, {"pixel": 0.5})
        if search_range_and_tol_comm_z is None:
            search_range_and_tol_comm_z = ({"angle": 6.0}, {"pixel": 1.0})
        if search_range_and_tol_comm_y_golden is None:
            search_range_and_tol_comm_y_golden = ({"angle": 0.5}, {"pixel": 0.4})

        def parse_spec(spec: dict[str, float], angle: str) -> float:
            """Parses a specification dictionary and returns the value in degrees."""
            if "angle" in spec:
                if "pixel" in spec:
                    raise ValueError(f"Spec cannot have both 'angle' and 'pixel' keys: {spec}")
                return spec["angle"]
            elif "pixel" in spec:
                convert: Callable[[float], float] = {
                    "x": self.pixel_to_euler_x, "y": self.pixel_to_euler_y, "z": self.pixel_to_euler_z
                }[angle]
                return convert(spec["pixel"])
            else:
                raise ValueError(f"Spec must have either 'angle' or 'pixel' key: {spec}")

        # convert the search ranges and tolerances to degrees.
        search_range_diff_y, tol_diff_y = [parse_spec(s, "y") for s in search_range_and_tol_diff_y]
        search_range_comm_y, tol_comm_y = [parse_spec(s, "y") for s in search_range_and_tol_comm_y]
        search_range_diff_x, tol_diff_x = [parse_spec(s, "x") for s in search_range_and_tol_diff_x]
        search_range_diff_z, tol_diff_z = [parse_spec(s, "z") for s in search_range_and_tol_diff_z]
        search_range_comm_z, tol_comm_z = [parse_spec(s, "z") for s in search_range_and_tol_comm_z]
        search_range_comm_y_golden, tol_comm_y_golden = [parse_spec(s, "y") for s in search_range_and_tol_comm_y_golden]
        specs_dict = dict(
            search_range_and_tol_diff_y_degrees=(search_range_diff_y, tol_diff_y),
            search_range_and_tol_comm_y_degrees=(search_range_comm_y, tol_comm_y),
            search_range_and_tol_diff_x_degrees=(search_range_diff_x, tol_diff_x),
            search_range_and_tol_diff_z_degrees=(search_range_diff_z, tol_diff_z),
            search_range_and_tol_comm_z_degrees=(search_range_comm_z, tol_comm_z),
            search_range_and_tol_comm_y_golden_degrees=(search_range_comm_y_golden, tol_comm_y_golden),
        )
        if dry_run_for_spec:
            logger.info("Dry run for spec parsing. Returning without performing calibration.")
            self._display_search_spec(**specs_dict)
            return specs_dict

        return self.calibrate_with_angle_spec(initial_stereo_state=initial_stereo_state, **specs_dict)

    def calibrate_with_pixel_spec(
        self,
        initial_stereo_state: BaselineFrameStereoState,
        search_range_and_tol_diff_y_pixels: tuple[float, float] = (25.5, 4.25),
        search_range_and_tol_comm_y_pixels: tuple[float, float] = (0.0, 4.25),
        search_range_and_tol_diff_x_pixels: tuple[float, float] = (560.0, 0.5),
        search_range_and_tol_diff_z_pixels: tuple[float, float] = (150.0, 0.5),
        search_range_and_tol_comm_z_pixels: tuple[float, float] = (150.0, 1.0),
        search_range_and_tol_comm_y_golden_pixels: tuple[float, float] = (1.6, 0.4),
        dry_run_for_spec: bool = False,
    ) -> BaselineFrameStereoState | dict[str, tuple[float, float]]:
        """Performs the initial calibration.

        Args:
            initial_stereo_state: The initial stereo state which will be used as a pre-condition.
            search_range_and_tol_diff_y_pixels: The search range and tolerance for the differential euler y angle,
                in pixels. This is equivalent to the old terminology (2 * vib_y, tol_y).
            search_range_and_tol_comm_y_pixels: The search range and tolerance for the common euler y angle (formerly
                Tz), in degrees. This is used in the grid search of euler-y in addition to the differential euler y.
            search_range_and_tol_diff_x_pixels: The search range and tolerance for the differential euler x angle,
                in pixels. This is equivalent to the old terminology (2 * vib_x, tol_x).
            search_range_and_tol_diff_z_pixels: The search range and tolerance for the differential euler z angle,
                in pixels. This is equivalent to the old terminology (2 * vib_z, tol_z).
            search_range_and_tol_comm_z_pixels: The search range and tolerance for the common euler z angle (formerly
                Ty), in pixels. This is equivalent to the old terminology (max_angle_y - min_angle_y, acc_y).
            search_range_and_tol_comm_y_golden_pixels: The search range and tolerance for the common euler y angle
                (formerly Tz), in pixels. This is equivalent to the old terminology (max_angle_z - min_angle_z, acc_z).
            dry_run_for_spec: If True, do not perform the calibration, just return the parsed search ranges and
                tolerances. This is useful for testing and checking the specification parsing.
        """
        # convert the search ranges and tolerances to degrees.
        search_range_diff_y, tol_diff_y = [self.pixel_to_euler_y(p) for p in search_range_and_tol_diff_y_pixels]
        search_range_comm_y, tol_comm_y = [self.pixel_to_euler_y(p) for p in search_range_and_tol_comm_y_pixels]
        search_range_diff_x, tol_diff_x = [self.pixel_to_euler_x(p) for p in search_range_and_tol_diff_x_pixels]
        search_range_diff_z, tol_diff_z = [self.pixel_to_euler_z(p) for p in search_range_and_tol_diff_z_pixels]
        search_range_comm_z, tol_comm_z = [self.pixel_to_euler_z(p) for p in search_range_and_tol_comm_z_pixels]
        search_range_comm_y_golden, tol_comm_y_golden = [
            self.pixel_to_euler_y(p) for p in search_range_and_tol_comm_y_golden_pixels
        ]
        specs_dict = dict(
            search_range_and_tol_diff_y_degrees=(search_range_diff_y, tol_diff_y),
            search_range_and_tol_comm_y_degrees=(search_range_comm_y, tol_comm_y),
            search_range_and_tol_diff_x_degrees=(search_range_diff_x, tol_diff_x),
            search_range_and_tol_diff_z_degrees=(search_range_diff_z, tol_diff_z),
            search_range_and_tol_comm_z_degrees=(search_range_comm_z, tol_comm_z),
            search_range_and_tol_comm_y_golden_degrees=(search_range_comm_y_golden, tol_comm_y_golden),
        )
        if dry_run_for_spec:
            logger.info("Dry run for spec parsing. Returning without performing calibration.")
            self._display_search_spec(**specs_dict)
            return specs_dict
        return self.calibrate_with_angle_spec(initial_stereo_state=initial_stereo_state, **specs_dict)

    def calibrate_with_angle_spec(
        self,
        initial_stereo_state: BaselineFrameStereoState,
        search_range_and_tol_diff_y_degrees: tuple[float, float],
        search_range_and_tol_comm_y_degrees: tuple[float, float],
        search_range_and_tol_diff_x_degrees: tuple[float, float],
        search_range_and_tol_diff_z_degrees: tuple[float, float],
        search_range_and_tol_comm_z_degrees: tuple[float, float],
        search_range_and_tol_comm_y_golden_degrees: tuple[float, float],
        dry_run_for_spec: bool = False,
    ):
        """Performs the initial calibration.

        Args:
            initial_stereo_state: The initial stereo state which will be used as a pre-condition.
            search_range_and_tol_diff_y_degrees: The search range and tolerance for the differential euler y angle,
                in degrees. This is equivalent to the old terminology (2 * vib_y, tol_y).
            search_range_and_tol_comm_y_degrees: The search range and tolerance for the common euler y angle (formerly
                Tz), in degrees. This is used in the grid search of euler-y in addition to the differential euler y.
            search_range_and_tol_diff_x_degrees: The search range and tolerance for the differential euler x angle,
                in degrees. This is equivalent to the old terminology (2 * vib_x, tol_x).
            search_range_and_tol_diff_z_degrees: The search range and tolerance for the differential euler z angle,
                in degrees. This is equivalent to the old terminology (2 * vib_z, tol_z).
            search_range_and_tol_comm_z_degrees: The search range and tolerance for the common euler z angle (formerly
                Ty), in degrees. This is equivalent to the old terminology (max_angle_y - min_angle_y, acc_y).
            search_range_and_tol_comm_y_golden_degrees: The search range and tolerance for the common euler y angle
                (formerly Tz), in degrees. This is equivalent to the old terminology (max_angle_z - min_angle_z, acc_z).
                This is used in the Golden section search.
            dry_run_for_spec: If True, do not perform the calibration, just return the parsed search ranges and
                tolerances. This is useful for testing and checking the specification parsing.
        """
        specs_dict = dict(
            search_range_and_tol_diff_y_degrees=search_range_and_tol_diff_y_degrees,
            search_range_and_tol_comm_y_degrees=search_range_and_tol_comm_y_degrees,
            search_range_and_tol_diff_x_degrees=search_range_and_tol_diff_x_degrees,
            search_range_and_tol_diff_z_degrees=search_range_and_tol_diff_z_degrees,
            search_range_and_tol_comm_z_degrees=search_range_and_tol_comm_z_degrees,
            search_range_and_tol_comm_y_golden_degrees=search_range_and_tol_comm_y_golden_degrees,
        )
        if dry_run_for_spec:
            logger.info("Dry run for spec parsing. Returning without performing calibration.")
            self._display_search_spec(**specs_dict)
            return specs_dict

        if self.verbose:
            self._display_search_spec(**specs_dict)

        search_range_diff_y, tol_diff_y = search_range_and_tol_diff_y_degrees
        search_range_comm_y, tol_comm_y = search_range_and_tol_comm_y_degrees
        search_range_diff_x, tol_diff_x = search_range_and_tol_diff_x_degrees
        search_range_diff_z, tol_diff_z = search_range_and_tol_diff_z_degrees
        search_range_comm_z, tol_comm_z = search_range_and_tol_comm_z_degrees
        search_range_comm_y_golden, tol_comm_y_golden = search_range_and_tol_comm_y_golden_degrees

        initial_fom = self.fom_calc.calculate(self.image1_gpu, self.i1, self.image2_gpu, self.i2, initial_stereo_state)
        if self.verbose:
            logger.info(f"initial_fom: {initial_fom}")
        if self.debug_dir is not None:
            debug_weight = self.fom_calc.getWeight()
            np.save(self._debug_dir / "fom_calc_weight_map.npy", debug_weight)
            debug_rect_left, debug_rect_right, debug_disparity = self.fom_calc.getRectifiedImagesAndDisparity()
            cv.imwrite(str(self._debug_dir / "rectified_left.png"), debug_rect_left)
            cv.imwrite(str(self._debug_dir / "rectified_right.png"), debug_rect_right)
            np.save(self._debug_dir / "rectified_disparity.npy", debug_disparity)

        # Set up the optimizers.
        kwargs_images = dict(image1=self.image1_gpu, i1=self.i1, image2=self.image2_gpu, i2=self.i2)
        diff_x_optimizer = GoldenSectionEulerAngleOptimizer(
            **kwargs_images, angle_axis="x", common_angle=False, fom_calc=self.fom_x_roi
        )
        diff_z_optimizer = GoldenSectionEulerAngleOptimizer(
            **kwargs_images, angle_axis="z", common_angle=False, fom_calc=self.fom_z_roi
        )
        comm_y_optimizer = GoldenSectionEulerAngleOptimizer(
            **kwargs_images, angle_axis="y", common_angle=True, fom_calc=self.fom_calc
        )
        comm_z_optimizer = GoldenSectionEulerAngleOptimizer(
            **kwargs_images, angle_axis="z", common_angle=True, fom_calc=self.fom_calc
        )

        if self.verbose:
            logger.info("Search starts")

        # Set up search procedure for euler y.
        def search_euler_ys(curr_comm_y: float, curr_diff_y: float) -> tuple[float, BaselineFrameStereoState]:
            logger.info(f"Searching for (comm_y={curr_comm_y:.4e}, diff_y={curr_diff_y:.4e}) ...")
            base_state = (
                StateComposer.from_state(initial_stereo_state)
                .compose_euler_common("y", curr_comm_y, transform=True)
                .compose_euler_differential("y", curr_diff_y, transform=True)
            ).initial_state
            with contexttimer.Timer() as curr_timer:
                curr_state = self._search_in_euler_xz_space(
                    base_state, search_range_diff_x, search_range_diff_z, tol_diff_x, tol_diff_z
                )
                time_after_main_search = curr_timer.elapsed

                # Optimize x and z using 1D Golden section searches.
                curr_state = diff_x_optimizer.optimize(curr_state, search_range_diff_x, tol_diff_x)[0]
                curr_state = diff_z_optimizer.optimize(curr_state, search_range_diff_z, tol_diff_z)[0]

                # Evaluate the FOM for this state.
                curr_fom = self.fom_calc.calculate(self.image1_gpu, self.i1, self.image2_gpu, self.i2, curr_state)
            if self.verbose:
                logger.info(
                    f"Time for (comm_y={curr_comm_y:.4e}, diff_y={curr_diff_y:.4e}): "
                    f"{curr_timer.elapsed:.3f}s (main search: {time_after_main_search:.3f}s)"
                )
                logger.info(f"FOM for (comm_y={curr_comm_y:.4e}, diff_y={curr_diff_y:.4e}): {curr_fom}")
            return curr_fom, curr_state

        # Set up the grid spec in the search of euler y.
        tol_0, tol_1 = search_range_comm_y / 5, tol_comm_y
        if tol_0 <= tol_1:
            comm_y_grid_spec = [(search_range_comm_y, tol_1)]
        else:
            comm_y_grid_spec = [
                # First, do a coarse search of 5 points.
                (search_range_comm_y, tol_0),
                # Then, do a finer search within the 2 neighborhoods of the best point (2 * tol from previous round),
                # subtracting the two end points to avoid duplication (2 * tol of current round).
                (2 * tol_0 - 2 * tol_1, tol_1),
            ]
        diff_y_grid_to_search = self.get_search_grid(search_range_diff_y, tol_diff_y)

        # Search the common and differential euler y angles.
        y_search_result: list[tuple[float, BaselineFrameStereoState, float, float]] = []
        best_comm_y = 0.0  # will be updated in the loop below.
        for idx_try_comm_y, (search_range, tol) in enumerate(comm_y_grid_spec):
            # Expand around the best state found in the previous round.
            comm_y_grid_to_search = self.get_search_grid(search_range, tol, mid_point=best_comm_y)
            if idx_try_comm_y != 0:
                # Skip the best point of previous round because we already searched it.
                comm_y_grid_to_search = [y for y in comm_y_grid_to_search if y != best_comm_y]
            for comm_y in comm_y_grid_to_search:
                for diff_y in diff_y_grid_to_search:
                    fom, state = search_euler_ys(comm_y, diff_y)  # warm up
                    y_search_result.append((fom, state, comm_y, diff_y))
            # Collect the best state found in this round of common y search.
            y_search_result.sort(key=lambda tup: tup[0], reverse=True)  # sort according to fom.
            _, _, best_comm_y, _ = y_search_result[0]

        best_fom, best_state, best_comm_y, best_diff_y = y_search_result[0]
        if self.verbose:
            logger.info(f"Best y=(comm_y={best_comm_y}, diff_y={best_diff_y}): {best_fom}")
            logger.info(f"Best state: {best_state}")
        if self.debug_dir is not None:
            dumpable = [(t[0], np.array(t[1].as_list(), np.float64), t[2], t[3]) for t in y_search_result]
            with open(self._debug_dir / "debug_y_search_result.pickle", "wb") as f:
                pickle.dump(dumpable, f)

        # Refine the best state by searching common y and z.
        state = best_state
        with contexttimer.Timer() as timer:
            state = comm_y_optimizer.optimize(state, search_range_comm_y_golden, tol_comm_y_golden)[0]
            state = comm_z_optimizer.optimize(state, search_range_comm_z, tol_comm_z)[0]
        if self.verbose:
            logger.info(f"Time for common YZ search: {timer.elapsed:.3f}s")

        # Optimize x and z using 1D Golden section searches.
        state = diff_x_optimizer.optimize(state, search_range_diff_x, tol_diff_x)[0]
        state = diff_z_optimizer.optimize(state, search_range_diff_z, tol_diff_z)[0]

        final_fom = self.fom_calc.calculate(self.image1_gpu, self.i1, self.image2_gpu, self.i2, state)
        if self.verbose:
            logger.info(f"initial_fom: {initial_fom}")
            logger.info(f"final_fom:   {final_fom}")
            logger.info(f"initial state: {initial_stereo_state}")
            logger.info(f"final state:   {state}")
        return state

    def pixel_to_euler_x(self, pixel: float) -> float:
        """Returns the euler x angle in degrees corresponding to a pixel deviation."""
        return PixelDeviationToAngle.pixel_to_euler_x(pixel, self._rectified_focal)

    def pixel_to_euler_y(self, pixel: float) -> float:
        """Returns the euler y angle in degrees corresponding to a pixel deviation."""
        return PixelDeviationToAngle.pixel_to_euler_y(pixel, self._rectified_focal, self._rectified_size)

    def pixel_to_euler_z(self, pixel: float) -> float:
        """Returns the euler z angle in degrees corresponding to a pixel deviation."""
        return PixelDeviationToAngle.pixel_to_euler_z(pixel, self._rectified_size)

    def euler_x_to_pixel(self, angle_deg: float) -> float:
        """Returns the pixel deviation corresponding to an euler x angle in degrees."""
        return PixelDeviationToAngle.euler_x_to_pixel(angle_deg, self._rectified_focal)

    def euler_y_to_pixel(self, angle_deg: float) -> float:
        """Returns the pixel deviation corresponding to an euler y angle in degrees."""
        return PixelDeviationToAngle.euler_y_to_pixel(angle_deg, self._rectified_focal, self._rectified_size)

    def euler_z_to_pixel(self, angle_deg: float) -> float:
        """Returns the pixel deviation corresponding to an euler z angle in degrees."""
        return PixelDeviationToAngle.euler_z_to_pixel(angle_deg, self._rectified_size)

    @staticmethod
    def _display_search_spec(
        search_range_and_tol_diff_y_degrees,
        search_range_and_tol_comm_y_degrees,
        search_range_and_tol_diff_x_degrees,
        search_range_and_tol_diff_z_degrees,
        search_range_and_tol_comm_z_degrees,
        search_range_and_tol_comm_y_golden_degrees,
    ):
        search_range_diff_y, tol_diff_y = search_range_and_tol_diff_y_degrees
        search_range_comm_y, tol_comm_y = search_range_and_tol_comm_y_degrees
        search_range_diff_x, tol_diff_x = search_range_and_tol_diff_x_degrees
        search_range_diff_z, tol_diff_z = search_range_and_tol_diff_z_degrees
        search_range_comm_z, tol_comm_z = search_range_and_tol_comm_z_degrees
        search_range_comm_y_golden, tol_comm_y_golden = search_range_and_tol_comm_y_golden_degrees
        logger.info(f"              Search range (angle) | Search tol (angle)")
        logger.info(f"diff euler y: {search_range_diff_y: 20.6f} | {tol_diff_y: 18.6f}")
        logger.info(f"comm euler y: {search_range_comm_y: 20.6f} | {tol_comm_y: 18.6f}")
        logger.info(f"diff euler x: {search_range_diff_x: 20.6f} | {tol_diff_x: 18.6f}")
        logger.info(f"diff euler z: {search_range_diff_z: 20.6f} | {tol_diff_z: 18.6f}")
        logger.info(f"comm euler z: {search_range_comm_z: 20.6f} | {tol_comm_z: 18.6f}")
        logger.info(f"comm euler y: {search_range_comm_y_golden: 20.6f} | {tol_comm_y_golden: 18.6f} (golden section)")


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
            return StateComposer(r1, r2, s.t_norm).initial_state
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
