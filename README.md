# calibration-dev

Extrinsics calibration utilities for Nodar sensor workflows.

## Installation

```bash
pip install ".[pyhammer]"
# or, for development
pip install -e ".[runtime,pyhammer]"
pip install -e .  # for subsequent updates during development without installing slow-to-install packages.
```

## Usage

Basic example:

```python
from calibration import excal


# The following are example codes. The actual variables need to be created.
calibrator = excal.InitialCalibration(
    image_left,  # uint8, gray scale numpy array.
    intrinsic_left,  # pyhammer intrinsics instance.
    image_right,  # uint8, gray scale numpy array.
    intrinsic_right,  # pyhammer intrinsics instance.
    calib_planner,  # pyhammer rectification planner instance.
    (w, h),  # the rectified image's width and height.
)

with contexttimer.Timer() as t:
    calibrated_extrinsics = calibrator.calibrate(
        initial_stereo_state=extr, **calibrator.get_basic_factory_calibration_specs()
    )
print(t.elapsed)
```
