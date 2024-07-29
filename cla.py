import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


def process_image(
    file_path,
    oil_colour_hex="#878874",
    color_threshold=25,
    size_threshold=250,
    blur_kernel=(5, 5),
):
    image = cv2.imread(file_path)
    if image is None:
        return None, None, None

    oil_colour = tuple(int(oil_colour_hex[i : i + 2], 16) for i in (1, 3, 5))[::-1]
    oil_mask = np.sqrt(((image - oil_colour) ** 2).sum(axis=2)) < color_threshold
    oil_mask = np.uint8(oil_mask * 255)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    large_component_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_threshold:
            large_component_mask[labels == i] = 255
    edge_mask = cv2.GaussianBlur(large_component_mask, blur_kernel, 0)
    blurred_image = cv2.bitwise_and(image, image, mask=edge_mask)
    final_image = cv2.addWeighted(image, 0.7, blurred_image, 0.3, 0)

    oil_area = np.sum(large_component_mask == 255)
    total_area = image.shape[0] * image.shape[1]
    ratio = oil_area / total_area

    # Adding informative text to the image
    def put_text_with_background(img, text, position, font_scale, color, thickness):
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        box_coords = (
            (position[0], position[1] + 10),
            (position[0] + text_width, position[1] - text_height - 10),
        )
        cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Extract the time from the file name and position text
    time = file_path.split("=")[1].split("h")[0]
    put_text_with_background(
        image,
        f"Time: {time} hours",
        (image.shape[1] - 300, 50),
        1.0,
        (255, 255, 255),
        3,
    )
    put_text_with_background(
        final_image,
        f"Oil Area Ratio: {ratio:.2f}",
        (final_image.shape[1] - 350, 50),
        1.0,
        (255, 255, 255),
        3,
    )

    # Label for Original and Segmented Image
    put_text_with_background(image, "Original Image", (10, 50), 1.0, (255, 255, 255), 3)
    put_text_with_background(
        final_image, "Segmented Image", (10, 50), 1.0, (255, 255, 255), 3
    )

    # Combine original and processed images side by side
    combined = np.hstack((image, final_image))

    # Get total number of pixels in mask (not 0)
    total_area = np.sum(large_component_mask > 0)
    # Apply mask to image
    image[large_component_mask == 0] = 0
    # Get number of pixels in mask that are not 0
    oil_area = 1 - np.sum(image > 0) / (3 * total_area)

    # open the file errors.txt and append the error, and the number of images
    with open("errors.txt", "r") as f:
        data = f.read()
        if data:
            error, count = data.split(",")
            count = int(count) + 1
    with open("errors.txt", "w") as f:
        f.write(f"{float(error) + oil_area},{count}")

    return ratio, combined, image


def main(folder, input_folder, output_folder):
    ratios = []
    filenames = []

    i = 1
    while True:
        file_name = os.path.join(input_folder, f"T={i}h.jpg")
        if not os.path.exists(file_name):
            break

        ratio, combined_image, _ = process_image(file_name)
        if combined_image is None:
            break

        combined_filename = os.path.join(output_folder, f"combined_{i}.jpg")
        cv2.imwrite(combined_filename, combined_image)
        filenames.append(combined_filename)
        ratios.append(ratio)
        i += 1

    with imageio.get_writer(
        os.path.join(output_folder, "segmentation_side_by_side.gif"), mode="I", fps=1
    ) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    with imageio.get_writer(
        os.path.join(f"out_img/segmentation_side_by_side_{folder}.gif"), mode="I", fps=1
    ) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    hours = np.arange(1, len(ratios) + 1)

    try:
        # Exponential Fit
        popt, _ = curve_fit(exponential_decay, hours, ratios, p0=(1, 1, 0.1))
        predictions = exponential_decay(hours, *popt)
        label = "Exponential Decay"
    except:
        # Linear Regression
        X = hours.reshape(-1, 1)  # Feature matrix for linear regression
        model = LinearRegression()
        model.fit(X, ratios)
        predictions = model.predict(X)
        label = "Linear Regression"

    # Plotting the graph
    plt.figure(figsize=(10, 5))
    plt.plot(hours, ratios, marker="o", label="Actual Ratios")

    # Now evaluate the model against the data. If it performs worse than the threshold then do no plot
    # Calculate the mean squared error
    mse = np.mean((ratios - predictions) ** 2)
    if mse > 0.01:
        plt.plot(hours, predictions, label=label)

    plt.title(f"Oil Area Ratio Over Time with Trend Lines (Sample {folder})")
    plt.xlabel("Time (hours)")
    plt.ylabel("Oil to Total Area Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "ratio_plot.png"))
    plt.savefig(os.path.join(f"out_img/ratio_plot_{folder}.png"))
    # plt.show()
    
    # Write a txt file with the ratios
    with open(os.path.join(output_folder, "ratios.txt"), "w") as f:
        for i, ratio in enumerate(ratios):
            f.write(f"{hours[i]}h: {ratio:.4f}\n")
    # Write another file in the out_img folder
    with open(os.path.join(f"out_img/ratios_{folder}.txt"), "w") as f:
        for i, ratio in enumerate(ratios):
            f.write(f"{hours[i]}h: {ratio:.4f}\n")

    # Calculate the initial ratio
    initial_ratio = ratios[0]

    # Calculate recovery rate
    recovery_rates = [(initial_ratio - r) / initial_ratio * 100 for r in ratios]

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(hours, recovery_rates, "s-", color="tab:red", label="Recovery Rate (%)")
    plt.title(f"Recovery Rate Over Time (Sample {folder})")
    plt.xlabel("Time (hours)")
    plt.ylabel("Recovery Rate (%)")
    plt.grid(True)
    plt.xticks(hours)  # Ensure every hour is marked for clarity
    plt.legend()
    plt.savefig(os.path.join(output_folder, "recovery_rate_plot.png"))
    plt.savefig(os.path.join(f"out_img/recovery_rate_plot_{folder}.png"))
    # plt.show()
    
    # Write a txt file with the recovery rates
    with open(os.path.join(output_folder, "recovery_rates.txt"), "w") as f:
        for i, rate in enumerate(recovery_rates):
            f.write(f"{hours[i]}h: {rate:.4f}%\n")
    # Write another file in the out_img folder
    with open(os.path.join(f"out_img/recovery_rates_{folder}.txt"), "w") as f:
        for i, rate in enumerate(recovery_rates):
            f.write(f"{hours[i]}h: {rate:.4f}%\n")

# Function to model exponential decay
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


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

    # If output folder does not exist, create it
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # If output folder does not exist, create it
    if not os.path.exists("out_img"):
        os.makedirs("out_img")

    main(args.folder, args.input_folder, args.output_folder)
