# coding: utf-8

"""
This script takes an input image and either crops it or pads it to fit a given size.

The padding color is the average color of the image. Pasting a small image on a large image
uses seamlessClone to limit boundary artifacts.
"""

# Standard imports
import argparse

# External imports
import cv2
import numpy as np


def convert_to_square(image, size):
    final_image = np.zeros((size, size, 3), dtype=np.uint8)
    # Fill the image with the mean color
    mean_color = image.mean()
    final_image[:, :] = mean_color

    image = np.flip(image, axis=0)  # Flip the image horizontally

    mask = np.full_like(image, 255, dtype=np.uint8)

    final_image = cv2.seamlessClone(
        image,
        final_image,
        mask,
        (size // 2, size // 2),  # Center of the final image
        cv2.MIXED_CLONE,
    )

    return final_image


def convert_to_gray(image):
    """Convert an image to grayscale."""
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is colored
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # If already grayscale, return as is


def pad_crop_image(
    input_image, output_path, src_scale: float, tgt_size: int, tgt_scale: float
):
    input_img = cv2.imread(input_image)

    # --- Resize image to match target scale ---
    # Calculate resize factor: tgt_scale / src_scale (so that 1 micron in src matches tgt_scale pixels)
    resize_factor = src_scale / tgt_scale
    if resize_factor != 1.0:
        new_size = (
            int(round(input_img.shape[1] * resize_factor)),
            int(round(input_img.shape[0] * resize_factor)),
        )
        input_img = cv2.resize(input_img, new_size, interpolation=cv2.INTER_AREA)

    # Ensure the longest dimension is always vertical
    if input_img.shape[0] < input_img.shape[1]:
        input_img = np.transpose(input_img, (1, 0, 2))

    # Center crop a size x size part of the image if it is larger than size
    if input_img.shape[0] > tgt_size or input_img.shape[1] > tgt_size:
        start_y = max(0, (input_img.shape[0] - tgt_size) // 2)
        end_y = min(input_img.shape[0], start_y + tgt_size)

        start_x = max(0, (input_img.shape[1] - tgt_size) // 2)
        end_x = min(input_img.shape[1], start_x + tgt_size)
        input_img = input_img[start_y:end_y, start_x:end_x]

    target_img = convert_to_square(input_img, tgt_size)

    # Convert to grayscale if the input image is colored
    target_img = convert_to_gray(target_img)

    cv2.imwrite(str(output_path), target_img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Crop or pad an image to fit a given size."
    )
    parser.add_argument("input", type=str, help="Input image file path.")
    parser.add_argument("output", type=str, help="Output image file path.")
    parser.add_argument(
        "src_scale", type=float, help="Scale of the source image, in micron/pix."
    )
    parser.add_argument("tgt_size", type=int, help="Desired width of the output image.")
    parser.add_argument("tgt_scale", type=float, help="Scale in micron/pix")
    args = parser.parse_args()

    pad_crop_image(
        args.input, args.output, args.src_scale, args.tgt_size, args.tgt_scale
    )
