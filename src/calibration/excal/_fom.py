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
from pyhammer.cpyhammer import FomCalculator
from pyhammer.rectification import windowed_planner_wrap
from pyhammer.trinsics import BaselineFrameStereoState
from pyhammer.trinsics import IntrinsicsBase
from scipy.spatial.transform import Rotation

import calibration.optimiz as optimiz
import calibration.excal.specs as specs
from calibration.excal.specs import SpecValue


logger = logging.getLogger(__name__)
__all__ = ["StateComposer", "InitialCalibration"]
GRID_EVAL_OUTPUT_TYPE = list[tuple[float, tuple[float, float]]]


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


class XZStateComposer:
    def __init__(self, base_state: BaselineFrameStereoState):
        self.state_composer = StateComposer.from_state(base_state)

    def state_from_angles(self, euler_x_deg: float, euler_z_deg: float) -> BaselineFrameStereoState:
        composer = self.state_composer
        composer = composer.compose_euler_differential("x", euler_x_deg, transform=True)
        state = composer.compose_euler_differential("z", euler_z_deg)
        return state


class MultiscaleGridSearchXZOptimizer:
    """A class that performs 2D grid search to optimize euler x and z angles.

    This is derived from the GridSearchOptimizer class in the old codebase.

    Args:
        image1: The first image in the stereo pair.
        i1: The intrinsics of the first camera.
        image2: The second image in the stereo pair.
        i2: The intrinsics of the second camera.
        planner: The planner to use for rectification.
        specified_rectified_size: The size of the desired rectified images fed to the planner.plan(). If None, use the
            input image size. The actual rectified image size will be determined by the planner.
        total_search_disparity:  The total number of disparities to search over.
        match_border: The border directive when calculating FOM. If None, use strict padding.
        roi_directive: A string specifying the ROIs when calculating FOM. Valid values are:
            "horizontal": optimize for horizontal stereo, i.e., left-right stereo.
            "vertical": optimize for vertical stereo, i.e., top-bottom stereo.
            "center": Using a smaller central ROI.
            "large_center": Using a larger central ROI.
            "full": Using the full image.
        fom_weight_method:  An option to select which weight method to use. 0: no weighting, 1: weight by solid angle.
            This is passed to FomCalculator so look up the documentation in C++ there for details.
        verbose: If True, print verbose information during optimization.
    """
    def __init__(
        self,
        image1: npt.NDArray[np.uint8] | pyhammer.cpyhammer.cv_GpuMat,
        i1: IntrinsicsBase,
        image2: npt.NDArray[np.uint8],
        i2: IntrinsicsBase,
        planner: pyhammer.cpyhammer.AbstractPlanner,
        specified_rectified_size: tuple[int, int] | None = None,
        total_search_disparity: int = 256,
        match_border: pyhammer.cpyhammer.FomRoiBorderDirective | None = None,
        roi_directive: str = "horizontal",
        fom_weight_method: int = 0,
        verbose: bool = False,
    ):
        # The arguments in the initializer sets up the search landscape.
        self.image1_gpu = pyhammer.gpu_mat(image1)
        self.image2_gpu = pyhammer.gpu_mat(image2)
        self.i1 = i1
        self.i2 = i2
        self.planner = planner
        self.specified_rectified_size = specified_rectified_size
        self.total_search_disparity = total_search_disparity
        if match_border is not None:
            self.match_border = match_border
        else:
            self.match_border = pyhammer.cpyhammer.FomRoiBorderDirective()
            self.match_border.strict_padding = True
        self.roi_directive = roi_directive
        self.fom_weight_method = fom_weight_method
        self.verbose = verbose

        self._state_composer: XZStateComposer | None = None
        self._grid_search_best_state: tuple[float, float] | None = None
        self._grid_search_evaluations: GRID_EVAL_OUTPUT_TYPE | None = None

    def _reset_search_result(self) -> None:
        self._grid_search_best_state = (0.0, 0.0)
        self._grid_search_evaluations = []

    def _update_search_result(
        self,
        descending_grid_evals: GRID_EVAL_OUTPUT_TYPE,
        reference_score_func: Callable[[float, float], float],
    ) -> None:
        best_grid_state: tuple[float, float]
        _, best_grid_state = descending_grid_evals[0]  # descending order
        best_prev_grid_state: tuple[float, float] = self._grid_search_best_state
        if best_grid_state == best_prev_grid_state:
            # Always update to the latest given grid if the best states are the same.
            self._grid_search_best_state = best_grid_state
            self._grid_search_evaluations = descending_grid_evals
        else:
            # Use the current-scale FoM calculator to evaluate the states to decide which to use as the base.
            fom_from_grid = reference_score_func(*best_grid_state)
            fom_from_prev_best = reference_score_func(*best_prev_grid_state)
            if fom_from_grid >= fom_from_prev_best:  # The = sign is important. Update to the new grid when equal.
                self._grid_search_best_state = best_grid_state
                self._grid_search_evaluations = descending_grid_evals

    def last_sorted_grid_evaluations(self) -> GRID_EVAL_OUTPUT_TYPE:
        """Returns the last accepted grid evaluations.

        If the best result at a scale is not accepted, it means that the angles does not yield a better figure-of-merit
        according to a reference FoM calculation. The grid evaluation at this scale is then discarded. This function
        will only return the last accepted grid evaluations.
        """
        if self._grid_search_evaluations is None:
            return []
        return self._grid_search_evaluations

    def optimize(
        self,
        init_state: BaselineFrameStereoState | StateComposer,
        search_range_x_deg: float,
        search_step_x_deg: float,
        search_range_z_deg: float,
        search_step_z_deg: float,
    ) -> tuple[BaselineFrameStereoState, tuple[float, float]]:
        """Returns the optimized state and the corresponding FOM value."""
        window_spec = dict(window_center_x_offset_ratio=0.0, window_center_y_offset_ratio=0.0)
        fom_calc_kwargs = dict(
            border=self.match_border,
            fom_weight_method=self.fom_weight_method,
            initial_rectified_size=self.specified_rectified_size,
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
        elif self.roi_directive == "full":
            window_spec["window_w_ratio"] = 1.0
            window_spec["window_h_ratio"] = 1.0
        else:
            raise NotImplementedError(f"roi_directive={self.roi_directive} is not implemented.")
        # fom_calc in merit_helper will be set later.
        # merit_helper = XZSearchHelper(
        #     self.image1_gpu, self.i1, self.image2_gpu, self.i2, fom_calc=None, base_state=base_state
        # )
        self._state_composer = XZStateComposer(init_state)
        self._reset_search_result()

        # Performs the multiscale grid search.
        # In old codebase:
        #     _grid_search(x0, 4*delta_x, 4*delta_z,      vib_x,      vib_z, 3, self_max_disparity/4)
        #     _grid_search(x0, 2*delta_x, 2*delta_z,  delta_x*4,  delta_z*4, 2, self_max_disparity/4)
        #     _grid_search(x0,   delta_x,   delta_z, delta_x(!), delta_z(!), 1, self_max_disparity/2)  # see note (!)
        # note(herbert): The ranges in the last iteration in the following codes differ from the old code. I believe
        #                it is a bug in the old codebase. A `vib` of just 1*delta_x in the old codebase, which has an
        #                equivalent range of 2*delta_x, is too small to cover the segment spanned by a center point and
        #                its two neighbors.
        args = [
            (3, self.total_search_disparity // 4, 4 * search_step_x_deg, 4 * search_step_z_deg),
            (2, self.total_search_disparity // 4, 2 * search_step_x_deg, 2 * search_step_z_deg),
            (1, self.total_search_disparity // 2, 1 * search_step_x_deg, 1 * search_step_z_deg),
        ]
        x_range = search_range_x_deg
        z_range = search_range_z_deg
        grid_evaluations: GRID_EVAL_OUTPUT_TYPE = []
        for pyramid, search_disp, x_step, z_step in args:
            if self.verbose:
                logger.info(f"Searching Euler XZ grid for pyramid {pyramid}")
            # Set up the merit function.
            curr_scale_planner = windowed_planner_wrap(
                self.planner, **window_spec, post_windowing_scale=1 / 2 ** pyramid
            )
            curr_scale_fom_calc = FomCalculator.create(
                curr_scale_planner, **fom_calc_kwargs, matcher_num_disparities=search_disp,
            )
            def merit_func(x_deg: float, z_deg: float) -> float:
                s = self._state_composer.state_from_angles(x_deg, z_deg)
                return curr_scale_fom_calc.calculate(self.image1_gpu, self.i1, self.image2_gpu, self.i2, s)

            if grid_evaluations:
                self._update_search_result(grid_evaluations, reference_score_func=merit_func)
            # Perform the grid search.
            center = self._grid_search_best_state
            x_grid = get_search_grid(x_range, x_step, center[0])
            z_grid = get_search_grid(z_range, z_step, center[1])
            grid_evaluations = self.eval_grid2d(merit_func, x_grid, z_grid, verbose=self.verbose)
            grid_evaluations.sort(key=lambda x: x[0], reverse=True)
            # Update the search range for next scale.
            x_range = 2 * x_step
            z_range = 2 * z_step
            del curr_scale_planner, curr_scale_fom_calc, merit_func

        # Final update and return.
        fom_calc = FomCalculator.create(
            self.planner, **fom_calc_kwargs, matcher_num_disparities=self.total_search_disparity,
        )

        def merit_func(x_deg: float, z_deg: float) -> float:
            s = self._state_composer.state_from_angles(x_deg, z_deg)
            return fom_calc.calculate(self.image1_gpu, self.i1, self.image2_gpu, self.i2, s)

        self._update_search_result(grid_evaluations, merit_func)
        best_angles = self._grid_search_best_state
        return self._state_composer.state_from_angles(*best_angles), best_angles

    @staticmethod
    def eval_grid2d(
        func: Callable[[float, float], float], grid_0: Sequence[float], grid_1: Sequence[float], verbose: bool = False
    ) -> GRID_EVAL_OUTPUT_TYPE:
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
        self.angle_axis: str = angle_axis
        self.common_angle: bool = common_angle
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
        specified_rectified_size: The size of the desired rectified images fed to the planner.plan(). If None, use the
            input image size. The actual rectified image size will be determined by the planner.
        roi_directive: A string specifying the ROIs when calculating FOM. Valid values are:
            "horizontal": optimize for horizontal stereo, i.e., left-right stereo.
            "vertical": optimize for vertical stereo, i.e., top-bottom stereo.
            "center": Using a smaller central ROI.
            "large_center": Using a larger central ROI.
            "full": Using the full image.
        use_fom_weight: If True, use FOM weight when calculating FOM.
        verbose: If True, print verbose information during optimization.
        debug_dir: If provided, the directory to save debug information.
    """
    def __init__(
        self,
        image1: npt.NDArray[np.uint8],
        i1: IntrinsicsBase,
        image2: npt.NDArray[np.uint8],
        i2: IntrinsicsBase,
        planner: pyhammer.cpyhammer.AbstractPlanner,
        specified_rectified_size: tuple[int, int] | None = None,
        roi_directive: str = "horizontal",
        use_fom_weight: bool = False,
        verbose: bool = False,
        debug_dir: str | Path | None = None,
    ):
        self.verbose = verbose
        self._debug_dir = None
        self.debug_dir = debug_dir
        self._fom_weight_method = (1 if use_fom_weight else 0)
        self.image1_gpu = pyhammer.gpu_mat(image1)
        self.image2_gpu = pyhammer.gpu_mat(image2)
        self.i1 = i1
        self.i2 = i2
        self.planner = planner
        input_size = image1.shape[:2][::-1]
        if specified_rectified_size is None:
            specified_rectified_size = input_size
        self._specified_rectified_size = specified_rectified_size
        dummy_state = BaselineFrameStereoState(*([0.0] * 6), t_norm=1.0)
        temp_plan = planner.plan(i1, i2, input_size, dummy_state, specified_rectified_size)
        self._rectified_focal: float = temp_plan.intrinsic1.fx
        self._rectified_size: tuple[int, int] = temp_plan.output_size

        # TODO: border is only used when an ROI is specified!
        self.match_border = pyhammer.cpyhammer.FomRoiBorderDirective()
        # border.border = (9 - 1) // 2  # 9 is the matcher patch size
        self.match_border.strict_padding = True

        fom_calc_kwargs = dict(
            border=self.match_border,
            fom_weight_method=self._fom_weight_method,
            initial_rectified_size=specified_rectified_size,
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
        elif roi_directive == "full":
            self.x_roi_planners = [planner]
            self.z_roi_planners = [planner]
        else:
            raise NotImplementedError(f"roi_directive={roi_directive} is not implemented.")
        self.fom_x_roi = [FomCalculator.create(p, **fom_calc_kwargs) for p in self.x_roi_planners]
        self.fom_z_roi = [FomCalculator.create(p, **fom_calc_kwargs) for p in self.z_roi_planners]

    @staticmethod
    def get_basic_refinement_specs() -> dict[str, tuple[SpecValue, SpecValue]]:
        """Returns the basic refinement specifications for calibrate().

        This is based on Hammerhead specs:
            calibration_params.delta_x = 0.468509;  // unit: pixels
            calibration_params.delta_y = 4.37411;  // unit:pixels
            calibration_params.delta_z = 0.251327;  // unit: pixels
            calibration_params.vib_x = 46.8509;  // unit: pixels
            calibration_params.vib_y = 0;  // unit: pixels
            calibration_params.vib_z = 12.5664;  // unit: pixels

            // angle from Ty. This is a roll angle.
            calibration_params.min_angle_y = -25.1327;  // unit: pixels
            calibration_params.max_angle_y = 25.1327;  // unit: pixels
            calibration_params.acc_y = 1.00531;  // unit: pixels

            // Angle from Tz. This is a yaw angle.
            calibration_params.min_angle_z = -0.869913;  // unit: pixels
            calibration_params.max_angle_z = 0.871543;  // unit: pixels
            calibration_params.acc_z = 0.435567;  // unit: pixels
        """
        return dict(
            search_range_and_tol_diff_y=(SpecValue(0.000000000, "p"), SpecValue(4.374110, "p")),
            search_range_and_tol_comm_y=(SpecValue(0.000000000, "p"), SpecValue(4.374110, "p")),
            search_range_and_tol_diff_x=(SpecValue(2 * 46.8509, "p"), SpecValue(0.468509, "p")),
            search_range_and_tol_diff_z=(SpecValue(2 * 12.5664, "p"), SpecValue(0.251327, "p")),
            search_range_and_tol_comm_z=(SpecValue(25.1327 - -25.1327, "p"), SpecValue(1.00531, "p")),
            search_range_and_tol_comm_y_golden=(SpecValue(0.871543 - -0.869913, "p"), SpecValue(0.435567, "p")),
        )

    @staticmethod
    def get_basic_factory_calibration_specs():
        """Returns the basic factory calibration specifications for calibrate().

        This is based on Hammerhead specs:
            calibration_params.delta_x = 0.937018;  // unit: pixels
            calibration_params.delta_y = 4.37411;  // unit:pixels
            calibration_params.delta_z = 0.502655;  // unit: pixels
            calibration_params.vib_x = 281.106;  // unit: pixels
            calibration_params.vib_y = 13.2469;  // unit: pixels
            calibration_params.vib_z = 75.3982;  // unit: pixels

            // angle from Ty. This is a roll angle.
            calibration_params.min_angle_y = -75.3982;  // unit: pixels
            calibration_params.max_angle_y = 75.3982;  // unit: pixels
            calibration_params.acc_y = 1.00531;  // unit: pixels

            // Angle from Tz. This is a yaw angle.
            calibration_params.min_angle_z = -0.869913;  // unit: pixels
            calibration_params.max_angle_z = 0.871543;  // unit: pixels
            calibration_params.acc_z = 0.435567;  // unit: pixels
        """
        return dict(
            search_range_and_tol_diff_y=(SpecValue(2 * 13.2469, "p"), SpecValue(4.374110, "p")),
            search_range_and_tol_comm_y=(SpecValue(0.000000000, "p"), SpecValue(4.374110, "p")),
            search_range_and_tol_diff_x=(SpecValue(2 * 281.106, "p"), SpecValue(0.937018, "p")),
            search_range_and_tol_diff_z=(SpecValue(2 * 75.3982, "p"), SpecValue(0.502655, "p")),
            search_range_and_tol_comm_z=(SpecValue(75.3982 - -75.3982, "p"), SpecValue(1.00531, "p")),
            search_range_and_tol_comm_y_golden=(SpecValue(0.871543 - -0.869913, "p"), SpecValue(0.435567, "p")),
        )

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
        optimizer = MultiscaleGridSearchXZOptimizer(
            self.image1_gpu,
            self.i1,
            self.image2_gpu,
            self.i2,
            self.planner,
            self._specified_rectified_size,
            total_search_disparity=total_search_disparity,
            match_border=self.match_border,
            roi_directive=self.roi_directive,
            fom_weight_method=self._fom_weight_method,
            verbose=self.verbose,
        )
        _, best_state_angles = optimizer.optimize(
            base_state, search_range_x_deg, search_step_x_deg, search_range_z_deg, search_step_z_deg,
        )

        # Prepare computation object and the initial state for the next stage of the search.
        state_composer = XZStateComposer(base_state)
        def merit_func(x_deg: float, z_deg: float) -> float:
            s = state_composer.state_from_angles(x_deg, z_deg)
            return self.fom_calc.calculate(self.image1_gpu, self.i1, self.image2_gpu, self.i2, s)

        # Build a simplex for N-M algorithm
        initial_simplex: list[tuple[float, float]] = []
        grid_evaluations = optimizer.last_sorted_grid_evaluations()
        if grid_evaluations and merit_func(0.0, 0.0) <= merit_func(*best_state_angles):
            # The grid search estimate is better than the initial base state.
            if self.verbose:
                logger.info("Create simplex from the grid search result.")
            simplex_tol = search_step_x_deg * search_step_z_deg / 65
            _, p0 = grid_evaluations[0]
            _, p1 = grid_evaluations[1]
            p2 = None
            # Find the first point that is not collinear with p0 and p1, starting from the points with large FoM.
            for _, point in grid_evaluations[2:]:
                if 0.5 * abs(self.double_triangle_area(p0, p1, point)) > simplex_tol:
                    p2 = point
                    break
            assert p2 is not None
            # Shrink the simplex a bit. (tested and gives better results for precise fine-tuning)
            initial_simplex.append(p0)
            initial_simplex.append((0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])))
            initial_simplex.append((0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1])))
        else:
            # The initial state is already the best.
            if self.verbose:
                logger.info("Create simplex from the initial state instead of the grid search result.")
            initial_simplex.append((0.0, 0.0))
            initial_simplex.append((0.5 * search_step_x_deg, 0.0))
            initial_simplex.append((0.0, 0.5 * search_step_z_deg))

        # Nelder-Mead algorithm
        f_nm = lambda p: -merit_func(p[0], p[1])
        nm_result = optimiz.nelder_mead(f_nm, initial_simplex=initial_simplex, max_iter=2)
        best_angles = nm_result["x"]
        return state_composer.state_from_angles(*best_angles)

    def calibrate(
        self,
        initial_stereo_state: BaselineFrameStereoState,
        search_range_and_tol_diff_y: tuple[SpecValue, SpecValue] = (SpecValue(6.0, "d"), SpecValue(4.25, "p")),
        search_range_and_tol_comm_y: tuple[SpecValue, SpecValue] = (SpecValue(0.0, "d"), SpecValue(4.25, "p")),
        search_range_and_tol_diff_x: tuple[SpecValue, SpecValue] = (SpecValue(6.0, "d"), SpecValue(0.5, "p")),
        search_range_and_tol_diff_z: tuple[SpecValue, SpecValue] = (SpecValue(6.0, "d"), SpecValue(0.5, "p")),
        search_range_and_tol_comm_z: tuple[SpecValue, SpecValue] = (SpecValue(6.0, "d"), SpecValue(1.0, "p")),
        search_range_and_tol_comm_y_golden: tuple[SpecValue, SpecValue] = (SpecValue(0.5, "d"), SpecValue(0.4, "p")),
        dry_run_for_spec: bool = False,
    ) -> BaselineFrameStereoState | dict[str, tuple[float, ...]]:
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
            search_range_and_tol_comm_y_golden: The search range and tolerance of golden section search for the common
                euler y angle (formerly Tz).
            dry_run_for_spec: If True, do not perform the calibration, just return the parsed search ranges and
                tolerances. This is useful for testing and checking the specification parsing.
        """
        def to_angle(spec: specs.SpecValue, angle_name: str) -> float:
            """Parses a specification and returns the value in degrees."""
            if spec.unit == "d":
                return spec.value
            elif spec.unit == "p":
                if angle_name == "x":
                    return self.pixel_to_euler_x(spec.value)
                elif angle_name == "y":
                    return self.pixel_to_euler_y(spec.value)
                elif angle_name == "z":
                    return self.pixel_to_euler_z(spec.value)
                else:
                    raise ValueError(f"Invalid angle name '{angle_name}'")
            else:
                raise ValueError(f"Spec unit must have either 'd' or 'p' key, got '{spec.unit}'")

        # convert the search ranges and tolerances to degrees.
        specs_dict = dict(
            search_range_and_tol_diff_y_degrees=tuple(to_angle(s, "y") for s in search_range_and_tol_diff_y),
            search_range_and_tol_comm_y_degrees=tuple(to_angle(s, "y") for s in search_range_and_tol_comm_y),
            search_range_and_tol_diff_x_degrees=tuple(to_angle(s, "x") for s in search_range_and_tol_diff_x),
            search_range_and_tol_diff_z_degrees=tuple(to_angle(s, "z") for s in search_range_and_tol_diff_z),
            search_range_and_tol_comm_z_degrees=tuple(to_angle(s, "z") for s in search_range_and_tol_comm_z),
            search_range_and_tol_comm_y_golden_degrees=tuple(
                to_angle(s, "y") for s in search_range_and_tol_comm_y_golden
            ),
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
        diff_y_grid_to_search = get_search_grid(search_range_diff_y, tol_diff_y)

        # Search the common and differential euler y angles.
        y_search_result: list[tuple[float, BaselineFrameStereoState, float, float]] = []
        best_comm_y = 0.0  # will be updated in the loop below.
        for idx_try_comm_y, (search_range, tol) in enumerate(comm_y_grid_spec):
            # Expand around the best state found in the previous round.
            comm_y_grid_to_search = get_search_grid(search_range, tol, mid_point=best_comm_y)
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
        return specs.pixel_to_euler_x(pixel, self._rectified_focal)

    def pixel_to_euler_y(self, pixel: float) -> float:
        """Returns the euler y angle in degrees corresponding to a pixel deviation."""
        return specs.pixel_to_euler_y(pixel, self._rectified_focal, self._rectified_size)

    def pixel_to_euler_z(self, pixel: float) -> float:
        """Returns the euler z angle in degrees corresponding to a pixel deviation."""
        return specs.pixel_to_euler_z(pixel, self._rectified_size)

    def euler_x_to_pixel(self, angle_deg: float) -> float:
        """Returns the pixel deviation corresponding to an euler x angle in degrees."""
        return specs.euler_x_to_pixel(angle_deg, self._rectified_focal)

    def euler_y_to_pixel(self, angle_deg: float) -> float:
        """Returns the pixel deviation corresponding to an euler y angle in degrees."""
        return specs.euler_y_to_pixel(angle_deg, self._rectified_focal, self._rectified_size)

    def euler_z_to_pixel(self, angle_deg: float) -> float:
        """Returns the pixel deviation corresponding to an euler z angle in degrees."""
        return specs.euler_z_to_pixel(angle_deg, self._rectified_size)

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
