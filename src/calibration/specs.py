"""A module that contains search specification related functions."""
import numpy as np


class PixelDeviationToAngle:
    """A class that contains static methods to convert row pixel deviations to euler angles in horizontal stereo.

    For a horizontal stereo, if the camera is misaligned by a small angle, a feature in the stereo rectified image will
    be on two different rows. This difference, which we called deviation, is what we want to convert to or from an
    euler angle.
    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
