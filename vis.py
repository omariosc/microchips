import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_image(
    file_path,
    oil_colour_hex="#878874",
    color_threshold=25,
    size_threshold=250,
    blur_kernel=(5, 5),
    save_steps=False,
):
    image = cv2.imread(file_path)
    if image is None:
        return None, None, None

    # Convert HEX oil colour to BGR and create initial oil mask
    oil_colour = tuple(int(oil_colour_hex[i : i + 2], 16) for i in (1, 3, 5))[::-1]
    oil_mask = np.sqrt(((image - oil_colour) ** 2).sum(axis=2)) < color_threshold
    oil_mask = np.uint8(oil_mask * 255)

    if save_steps:
        # convert mask to image
        oil_mask_image = cv2.merge((oil_mask, oil_mask, oil_mask))
        cv2.imwrite("visualisation/01_initial_mask.jpg", oil_mask_image)

    # Apply morphological opening to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    if save_steps:
        cleaned_image = cv2.merge((cleaned, cleaned, cleaned))
        cv2.imwrite("visualisation/02_cleaned_mask.jpg", cleaned_image)

    # Connected components to filter small objects
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    large_component_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_threshold:
            large_component_mask[labels == i] = 255

    if save_steps:
        large_component_image = cv2.merge(
            (large_component_mask, large_component_mask, large_component_mask)
        )
        # change all black to white
        large_component_image[large_component_image == 0] = 255
        cv2.imwrite("visualisation/03_large_component_mask.jpg", large_component_image)

    # Apply Gaussian blur to smooth the edges
    edge_mask = cv2.GaussianBlur(large_component_mask, blur_kernel, 0)
    if save_steps:
        cv2.imwrite("visualisation/04_edge_mask.jpg", edge_mask)  # Save edge mask

    blurred_image = cv2.bitwise_and(image, image, mask=edge_mask)
    final_image = cv2.addWeighted(image, 0.7, blurred_image, 0.3, 0)
    if save_steps:
        cv2.imwrite(
            "visualisation/05_final_image.jpg", final_image
        )  # Save final processed image

    oil_area = np.sum(large_component_mask == 255)
    total_area = image.shape[0] * image.shape[1]
    ratio = oil_area / total_area

    return ratio, final_image, image


def main(folder, input_folder, output_folder):
    ensure_dir("visualisation/")  # Ensure visualisation directory exists
    filenames = []
    ratios = []

    file_name = os.path.join(input_folder, "T=1h.jpg")  # Process only the first image
    if os.path.exists(file_name):
        ratio, combined_image, _ = process_image(file_name, save_steps=True)
        if combined_image is not None:
            combined_filename = os.path.join(output_folder, "combined_1.jpg")
            cv2.imwrite(combined_filename, combined_image)
            filenames.append(combined_filename)
            ratios.append(ratio)

    # Additional code to handle GIF creation, plotting, etc., goes here


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images to calculate oil recovery."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder containing the images to process.",
    )
    args = parser.parse_args()
    args.input_folder = f"img/{args.folder}"
    args.output_folder = f"out/{args.folder}"

    main(args.folder, args.input_folder, args.output_folder)
