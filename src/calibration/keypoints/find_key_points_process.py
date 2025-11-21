import argparse
import os

import cv2 as cv
import numpy as np
import omniglue


def two_tuple(s: str) -> tuple[int, int]:
    s.replace(" ", "")
    s = s.lstrip("(").rstrip(")")
    parts = s.split(",")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image0", type=str, required=True, help="Path to the input image0.")
    parser.add_argument("--image1", type=str, required=True, help="Path to the input image1.")
    parser.add_argument(
        "--tl0",
        type=two_tuple,
        required=True,
        help="The top-left corner in x,y for cropping image0, as a string."
    )
    parser.add_argument(
        "--tl1",
        type=two_tuple,
        required=True,
        help="The top-left corner in x,y for cropping image1, as a string."
    )
    parser.add_argument(
        "--crop_size",
        "-c",
        type=two_tuple,
        default=(300, 200),
        help="The width and height of the cropped image."
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save the output matches.")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Threshold for keypoint detection.")
    args = parser.parse_args()

    omniglue.omniglue_extract.MATCH_THRESHOLD = args.threshold

    # Load the images and convert to RGB
    image0 = cv.imread(args.image0)[..., ::-1]
    image1 = cv.imread(args.image1)[..., ::-1]

    # Sort out the crops
    crop_w, crop_h = args.crop_size
    tl0_x, tl0_y = args.tl0
    tl1_x, tl1_y = args.tl1
    slice0 = np.s_[tl0_y:tl0_y + crop_h, tl0_x:tl0_x + crop_w]
    slice1 = np.s_[tl1_y:tl1_y + crop_h, tl1_x:tl1_x + crop_w]

    # Find matches and sort by confidence
    model_base_path = os.path.join(os.path.dirname(__file__), "../models")
    og = omniglue.OmniGlue(
        og_export=os.path.join(model_base_path, "og_export"),
        sp_export=os.path.join(model_base_path, "sp_v6"),
        dino_export=os.path.join(model_base_path, "dinov2_vitb14_pretrain.pth"),
    )
    match_kp0s, match_kp1s, match_confidences = og.FindMatches(image0[slice0], image1[slice1])
    sorting_recipe = np.argsort(match_confidences)[::-1]  # Sort descending
    match_confidences = match_confidences[sorting_recipe]
    match_kp0s = match_kp0s[sorting_recipe]
    match_kp1s = match_kp1s[sorting_recipe]

    # Save in the original image coordinates
    match_kp0s_in_original = match_kp0s + np.array([[tl0_x, tl0_y]])
    match_kp1s_in_original = match_kp1s + np.array([[tl1_x, tl1_y]])
    result = np.array(
        [
            np.hstack([match_kp0s_in_original, match_confidences.astype(np.float64)[:, np.newaxis]]),
            np.hstack([match_kp1s_in_original, match_confidences.astype(np.float64)[:, np.newaxis]]),
        ]
    )
    # The dimension check
    # axis 0: the image index
    # axis 1: the match index
    # axis 2: (x, y, confidence)
    assert result.shape == (2, len(match_kp0s), 3)

    # remove obvious error points that happens at the boundary of the images
    invalid_pairs = (
        (result[0, :, 0] == tl0_x)  # left edge
        | (result[1, :, 0] == tl1_x)  # left edge
        | (result[0, :, 0] == tl0_x + crop_w - 1)  # right edge
        | (result[1, :, 0] == tl1_x + crop_w - 1)  # right edge
        | (result[0, :, 1] == tl0_y)  # top edge
        | (result[1, :, 1] == tl1_y)  # top edge
        | (result[0, :, 1] == tl0_y + crop_h - 1)  # bottom edge
        | (result[1, :, 1] == tl1_y + crop_h - 1)  # bottom edge
    )
    valid_pairs = ~invalid_pairs
    result = result[:, valid_pairs, :]

    output = args.output if args.output.endswith(".npy") else args.output + ".npy"
    np.save(output, result)


if __name__ == "__main__":
    main()
