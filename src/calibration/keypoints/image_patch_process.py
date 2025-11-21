"""The modules that handles making image patches for OmniGlue to find matched keypoints."""

import glob
import os
import subprocess

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class PatchMaker:
    def __init__(self, bar_config: str, undistorted_image_dir: str, match_pair_output_dir: str | None = None, use_mono=False):
        assert bar_config in ["A", "B"]
        self.bar_config = bar_config

        if bar_config == "A":
            # Map camera location to the camera id
            self.camera_id_map = {
                "left": "2",
                "right": "1",
                "top": "2",
                "bottom": "3",
            }
        else:
            # Map camera location to the camera id
            self.camera_id_map = {
                "left": "1",
                "right": "2",
                "top": "3",
                "bottom": "2",
            }

        # Map camera location to the sequence in a list
        self.camera_sequence_map = {k: int(v) - 1 for k, v in self.camera_id_map.items()}

        self.match_pair_output_dir = undistorted_image_dir if match_pair_output_dir is None else match_pair_output_dir

        image1_path = os.path.join(undistorted_image_dir, "cam1_mono.png" if use_mono else "cam1.png")
        image2_path = os.path.join(undistorted_image_dir, "cam2_mono.png" if use_mono else "cam2.png")
        image3_path = os.path.join(undistorted_image_dir, "cam3_mono.png" if use_mono else "cam3.png")
        self.image_path_sequence = [image1_path, image2_path, image3_path]

        image1 = cv.imread(image1_path)[..., ::-1]
        image2 = cv.imread(image2_path)[..., ::-1]
        image3 = cv.imread(image3_path)[..., ::-1]
        self.image_sequence = [image1, image2, image3]

        # Variable `patches` is [((top-left-1), (top-left-2), (top-left-3), (crop size))]
        # Each tuple is in (x, y) format. Use None for no patch in that camera.
        self.patches: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    def show_whole(self, figsize=(15, 5)):
        fh = plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.imshow(self.image_sequence[0])
        plt.title("Camera 1")
        plt.subplot(132)
        plt.imshow(self.image_sequence[1])
        plt.title("Camera 2")
        plt.subplot(133)
        plt.imshow(self.image_sequence[2])
        plt.title("Camera 3")
        plt.tight_layout()
        plt.show()
        return fh

    def show_patch(self, tl1_x, tl1_y, tl2_x, tl2_y, tl3_x, tl3_y, crop_w, crop_h, figsize=(15, 5)):
        # Print a tuple for an entry of `self.patches`
        print(f"    (({tl1_x},{tl1_y}), ({tl2_x},{tl2_y}), ({tl3_x},{tl3_y}), ({crop_w},{crop_h})),")
        slice1 = np.s_[tl1_y:tl1_y + crop_h, tl1_x:tl1_x + crop_w]
        slice2 = np.s_[tl2_y:tl2_y + crop_h, tl2_x:tl2_x + crop_w]
        slice3 = np.s_[tl3_y:tl3_y + crop_h, tl3_x:tl3_x + crop_w]
        img1 = self.image_sequence[0][slice1]
        img2 = self.image_sequence[1][slice2]
        img3 = self.image_sequence[2][slice3]
        fh = plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.imshow(img1)
        plt.title("Camera 1")
        plt.subplot(132)
        plt.imshow(img2)
        plt.title("Camera 2")
        plt.subplot(133)
        plt.imshow(img3)
        plt.title("Camera 3")
        plt.tight_layout()
        plt.show()
        return fh

    def show_patch2(self, tl1: tuple, tl2: tuple, tl3: tuple, crop_tuple: tuple, figsize=(15, 5)):
        tl1_x, tl1_y = tl1
        tl2_x, tl2_y = tl2
        tl3_x, tl3_y = tl3
        crop_w, crop_h = crop_tuple
        return self.show_patch(tl1_x, tl1_y, tl2_x, tl2_y, tl3_x, tl3_y, crop_w, crop_h, figsize)

    def set_patches(self, patches: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]]):
        self.patches = patches.copy()

    def dispatch(self):
        processes = []
        for i, job_spec in enumerate(self.patches):
            top_left_corners = job_spec[:3]
            crop = job_spec[3]

            # Horizontal stereo task
            tl0 = top_left_corners[self.camera_sequence_map["left"]]
            tl1 = top_left_corners[self.camera_sequence_map["right"]]
            if tl0 is not None and tl1 is not None:
                tl_cam2 = top_left_corners[1]
                output_name_extra = f"{i:03d}-{tl_cam2[0]}_{tl_cam2[1]}-{crop[0]}x{crop[1]}"
                output_name = (
                    f"{self.bar_config}-cam{self.camera_id_map['left']}cam{self.camera_id_map['right']}"
                    f"-{output_name_extra}.npy"
                )
                command = [
                    "python",
                    "find_key_points_process.py",
                    "--image0", self.image_path_sequence[self.camera_sequence_map["left"]],
                    "--image1", self.image_path_sequence[self.camera_sequence_map["right"]],
                    "--tl0", f"{tl0}".replace(" ", ""),
                    "--tl1", f"{tl1}".replace(" ", ""),
                    "--crop_size", f"{crop}".replace(" ", ""),
                    "--output", f"{self.match_pair_output_dir}/{output_name}",
                ]
                processes.append(subprocess.run(command))

            # Vertical stereo task
            tl0 = top_left_corners[self.camera_sequence_map["top"]]
            tl1 = top_left_corners[self.camera_sequence_map["bottom"]]
            if tl0 is not None and tl1 is not None:
                tl_cam2 = top_left_corners[1]
                output_name_extra = f"{i:03d}-{tl_cam2[0]}_{tl_cam2[1]}-{crop[0]}x{crop[1]}"
                output_name = (
                    f"{self.bar_config}-cam{self.camera_id_map['top']}cam{self.camera_id_map['bottom']}"
                    f"-{output_name_extra}.npy"
                )
                command = [
                    "python",
                    "find_key_points_process.py",
                    "--image0", self.image_path_sequence[self.camera_sequence_map["top"]],
                    "--image1", self.image_path_sequence[self.camera_sequence_map["bottom"]],
                    "--tl0", f"{tl0}".replace(" ", ""),
                    "--tl1", f"{tl1}".replace(" ", ""),
                    "--crop_size", f"{crop}".replace(" ", ""),
                    "--output", f"{self.match_pair_output_dir}/{output_name}",
                ]
                processes.append(subprocess.run(command))

    def inspect_matches(self, index: int, figsize=(16, 4)) -> plt.Figure | None:
        try:
            job_spec = self.patches[index]
            top_left_corners = job_spec[:3]
            crop = job_spec[3]
        except IndexError:
            print(f"Index {index} is out of range [0, {len(self.patches)})")
            return None

        match_point_set_h = glob.glob(
            self.match_pair_output_dir
            + f"/{self.bar_config}-cam{self.camera_id_map['left']}cam{self.camera_id_map['right']}-{index:03d}-*.npy"
        )
        match_point_set_v = glob.glob(
            self.match_pair_output_dir
            + f"/{self.bar_config}-cam{self.camera_id_map['top']}cam{self.camera_id_map['bottom']}-{index:03d}-*.npy"
        )
        if not match_point_set_h and not match_point_set_v:
            return None

        assert len(match_point_set_h) <= 1 and len(match_point_set_v) <= 1
        match_point_set_h = np.load(match_point_set_h[0]) if match_point_set_h else None
        match_point_set_v = np.load(match_point_set_v[0]) if match_point_set_v else None

        fh = plt.figure(figsize=figsize)
        if match_point_set_h is not None:
            xyc0, xyc1 = match_point_set_h
            ah = fh.add_subplot(141)
            ah.imshow(self.image_sequence[self.camera_sequence_map["left"]])
            for x, y, c in xyc0:
                ah.plot([x], [y], ".")
            tl_left = top_left_corners[self.camera_sequence_map["left"]]
            ah.set_xlim(tl_left[0], tl_left[0] + crop[0])
            ah.set_ylim(tl_left[1] + crop[1], tl_left[1])
            ah.set_title("Horizontal")

            ah = fh.add_subplot(142)
            ah.imshow(self.image_sequence[self.camera_sequence_map["right"]])
            for x, y, c in xyc1:
                ah.plot([x], [y], ".")
            tl_right = top_left_corners[self.camera_sequence_map["right"]]
            ah.set_xlim(tl_right[0], tl_right[0] + crop[0])
            ah.set_ylim(tl_right[1] + crop[1], tl_right[1])

        if match_point_set_v is not None:
            xyc0, xyc1 = match_point_set_v
            ah = fh.add_subplot(143)
            ah.imshow(self.image_sequence[self.camera_sequence_map["top"]])
            for x, y, c in xyc0:
                ah.plot([x], [y], ".")
            tl_top = top_left_corners[self.camera_sequence_map["top"]]
            ah.set_xlim(tl_top[0], tl_top[0] + crop[0])
            ah.set_ylim(tl_top[1] + crop[1], tl_top[1])
            ah.set_title("Vertical")

            ah = fh.add_subplot(144)
            ah.imshow(self.image_sequence[self.camera_sequence_map["bottom"]])
            for x, y, c in xyc1:
                ah.plot([x], [y], ".")
            tl_bottom = top_left_corners[self.camera_sequence_map["bottom"]]
            ah.set_xlim(tl_bottom[0], tl_bottom[0] + crop[0])
            ah.set_ylim(tl_bottom[1] + crop[1], tl_bottom[1])
        fh.tight_layout()
        return fh
