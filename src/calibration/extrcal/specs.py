"""A module that contains search specification related functions.

The functions convert row deviations in pixels to euler angles and vice versa for horizontal stereo cameras.  For a
horizontal stereo, if the camera is misaligned by a small angle, a feature in the stereo rectified image will be on two
different rows. This difference, which we called deviation, is what we want to convert to or from an euler angle.
"""
from numbers import Real
from typing import NamedTuple

import numpy as np


class SpecValue(NamedTuple):
    """A specification value with a value and a unit.

    The unit is a string that describes the unit of the value. Allowed values are:
    - "p" for pixels
    - "d" for degrees
    """
    value: float
    unit: str

    # ---------- helpers ----------
    def _assert_same_unit(self, other: "SpecValue", op: str) -> None:
        if self.unit != other.unit:
            raise ValueError(
                f"Cannot apply '{op}' to different units: {self.unit!r} and {other.unit!r}"
            )

    def _coerce_real(self, other: object, op: str) -> float:
        if not isinstance(other, Real):
            raise TypeError(f"Unsupported operand type(s) for {op}: 'SpecValue' and {type(other).__name__!r}")
        return float(other)

    # ---------- unary ----------
    def __neg__(self) -> "SpecValue":
        return SpecValue(-self.value, self.unit)

    def __pos__(self) -> "SpecValue":
        return self

    def __abs__(self) -> "SpecValue":
        return SpecValue(abs(self.value), self.unit)

    # ---------- addition / subtraction ----------
    def __add__(self, other: object) -> "SpecValue":
        if isinstance(other, SpecValue):
            self._assert_same_unit(other, "+")
            return SpecValue(self.value + other.value, self.unit)
        if isinstance(other, Real):
            return SpecValue(self.value + float(other), self.unit)
        return NotImplemented

    def __radd__(self, other: object) -> "SpecValue":
        return self.__add__(other)

    def __sub__(self, other: object) -> "SpecValue":
        if isinstance(other, SpecValue):
            self._assert_same_unit(other, "-")
            return SpecValue(self.value - other.value, self.unit)
        if isinstance(other, Real):
            return SpecValue(self.value - float(other), self.unit)
        return NotImplemented

    def __rsub__(self, other: object) -> "SpecValue":
        if isinstance(other, Real):
            return SpecValue(float(other) - self.value, self.unit)
        if isinstance(other, SpecValue):
            self._assert_same_unit(other, "-")
            return SpecValue(other.value - self.value, self.unit)
        return NotImplemented

    # ---------- multiplication ----------
    # Allow scalar multiplication only.
    def __mul__(self, other: object) -> "SpecValue":
        factor = self._coerce_real(other, "*")
        return SpecValue(self.value * factor, self.unit)

    def __rmul__(self, other: object) -> "SpecValue":
        return self.__mul__(other)

    # ---------- division ----------
    # SpecValue / scalar -> SpecValue
    # SpecValue / SpecValue (same unit) -> float ratio
    def __truediv__(self, other: object) -> "SpecValue | float":
        if isinstance(other, SpecValue):
            self._assert_same_unit(other, "/")
            if other.value == 0:
                raise ZeroDivisionError("division by zero SpecValue")
            return self.value / other.value

        divisor = self._coerce_real(other, "/")
        if divisor == 0:
            raise ZeroDivisionError("division by zero")
        return SpecValue(self.value / divisor, self.unit)

    # scalar / SpecValue -> float
    def __rtruediv__(self, other: object) -> float:
        if not isinstance(other, SpecValue):
            raise TypeError(f"Unsupported division with a scalar numerator and {self.__class__.__name__} denominator.")
        self._assert_same_unit(other, "/")
        if self.value == 0:
            raise ZeroDivisionError("division by zero SpecValue")
        return other.value / self.value

    # Optional convenience for numeric APIs
    def __float__(self) -> float:
        return float(self.value)


def spec_value(arg1: str | tuple[float, str] | SpecValue, arg2: str | None = None) -> SpecValue:
    """Creates a SpecValue instance.

    Args:
        arg1: If arg2 is None, arg1 is a SpecValue instance and is returned as is. If arg2 is not None, arg1 is the
            value of the SpecValue.
        arg2: The unit of the SpecValue.
    """
    value = None
    unit = None
    if arg2 is None:
        if isinstance(arg1, SpecValue):
            return arg1
        if isinstance(arg1, tuple):
            value, unit = arg1[0], arg1[1]
        elif isinstance(arg1, str):
            value, unit = arg1[:-1], arg1[-1]
    else:
        value = arg1
        unit = arg2
    value = float(value)
    if unit not in ("p", "d"):
        raise ValueError("Unit must be either 'p' for pixels or 'd' for degrees.")
    return SpecValue(value=value, unit=unit)


def pixel_to_euler_x(row_deviation_pixel: float, focal_length_pixel: float) -> float:
    """Converts a row pixel deviation to an euler x angle in degrees.

    Args:
        row_deviation_pixel: The row pixel deviation between the two images in the stereo rectified pair.
        focal_length_pixel: The focal length of the rectified image in pixels.
    """
    # pixelToEulerXPerturbation() in C++ codebase.
    opposite = row_deviation_pixel
    adjacent = focal_length_pixel
    return np.rad2deg(opposite / adjacent)  # use small angle approximation


def euler_x_to_pixel(euler_x_deg: float, focal_length_pixel: float) -> float:
    """Converts an euler x angle in degrees to a row pixel deviation.

    Args:
        euler_x_deg: The euler x angle in degrees.
        focal_length_pixel: The focal length of the rectified image in pixels.
    """
    # eulerXPerturbationToPixel() in C++ codebase.
    angle_rad = np.deg2rad(euler_x_deg)
    adjacent = focal_length_pixel
    return adjacent * angle_rad  # use small angle approximation


def pixel_to_euler_y(
    row_deviation_pixel: float, focal_length_pixel: float, image_size: tuple[float, float]
) -> float:
    """Converts a pixel deviation to an euler y angle in degrees.

    Args:
        row_deviation_pixel: The row pixel deviation between the two images in the stereo rectified pair.
        focal_length_pixel: The focal length of the rectified image in pixels.
        image_size: The size of the rectified image in pixels, as (width, height).
    """
    # pixelToEulerYPerturbation() in C++ codebase.
    # Use the extreme point as the testing point.
    x = image_size[0] / 2.0
    y = image_size[1] / 2.0
    z = focal_length_pixel
    # use small angle approximation
    return np.rad2deg(z * row_deviation_pixel / (x * (y + row_deviation_pixel)))


def euler_y_to_pixel(
    euler_y_deg: float, focal_length_pixel: float, image_size: tuple[float, float]
) -> float:
    """Converts an euler y angle in degrees to a row pixel deviation.

    Args:
        euler_y_deg: The euler y angle in degrees.
        focal_length_pixel: The focal length of the rectified image in pixels.
        image_size: The size of the rectified image in pixels, as (width, height).
    """
    # eulerYPerturbationToPixel() in C++ codebase.
    angle_rad = np.deg2rad(euler_y_deg)
    # Use the extreme point as the testing point.
    x = image_size[0] / 2.0
    y = image_size[1] / 2.0
    z = focal_length_pixel
    # use small angle approximation
    return (x * angle_rad) * y / (z - x * angle_rad)


def pixel_to_euler_z(row_deviation_pixel: float, image_size: tuple[float, float]) -> float:
    """Converts a pixel deviation to an euler z angle in degrees.

    Args:
        row_deviation_pixel: The pixel deviation between the two images in the stereo rectified pair.
        image_size: The size of the rectified image in pixels, as (width, height).
    """
    # pixelToEulerZPerturbation() in C++ codebase.
    # Use the extreme point as the testing point.
    opposite = row_deviation_pixel
    adjacent = image_size[0] / 2.0
    return np.rad2deg(opposite / adjacent)  # use small angle approximation


def euler_z_to_pixel(euler_z_deg: float, image_size: tuple[float, float]) -> float:
    """Converts an euler z angle in degrees to a pixel deviation.

    Args:
        euler_z_deg: The euler z angle in degrees.
        image_size: The size of the rectified image in pixels, as (width, height).
    """
    # eulerZPerturbationToPixel() in C++ codebase.
    angle_rad = np.deg2rad(euler_z_deg)
    adjacent = image_size[0] / 2.0
    return adjacent * angle_rad  # use small angle approximation
