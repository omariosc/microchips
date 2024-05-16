import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2
import os

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# global center_colors
# center_colors = {}

# def process_image(file_path, size_threshold=300):
#     # Load the image
#     image = cv2.imread(file_path)
#     if image is None:
#         return None, None, None

#     # Convert to grayscale and equalize
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(gray)

#     # Thresholding and inversion
#     _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     inverted_thresh = cv2.bitwise_not(thresh)

#     # Morphological cleaning on the inverted mask
#     kernel = np.ones((3, 3), np.uint8)
#     cleaned = cv2.morphologyEx(inverted_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

#     # Analyze connected components
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
#         cleaned, connectivity=8
#     )
#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_AREA] < size_threshold:
#             cleaned[labels == i] = 0

#     # Calculate the ratio of oil to total area
#     oil_area = np.sum(cleaned == 255)
#     total_area = image.shape[0] * image.shape[1]
#     ratio = oil_area / total_area

#     # Create overlay image with transparency
#     overlay = image.copy()
#     mask = cleaned == 255
#     overlay[mask] = (0, 255, 0)

#     # Prepare text with background boxes
#     def put_text_with_background(img, text, position, font_scale, color, thickness):
#         (text_width, text_height), _ = cv2.getTextSize(
#             text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
#         )
#         box_coords = (
#             (position[0], position[1] + 10),
#             (position[0] + text_width, position[1] - text_height - 10),
#         )
#         cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
#         cv2.putText(
#             img,
#             text,
#             position,
#             cv2.FONT_HERSHEY_SIMPLEX,
#             font_scale,
#             color,
#             thickness,
#             cv2.LINE_AA,
#         )

#     # Extract the time from the file name and position text
#     time = file_path.split("=")[1].split("h")[0]
#     put_text_with_background(
#         image,
#         f"Time: {time} hours",
#         (image.shape[1] - 300, 50),
#         1.0,
#         (255, 255, 255),
#         3,
#     )
#     put_text_with_background(
#         overlay,
#         f"Oil Area Ratio: {ratio:.2f}",
#         (overlay.shape[1] - 350, 50),
#         1.0,
#         (255, 255, 255),
#         3,
#     )

#     # Label for Original and Segmented Image
#     put_text_with_background(image, "Original Image", (10, 50), 1.0, (255, 255, 255), 3)
#     put_text_with_background(
#         overlay, "Segmented Image", (10, 50), 1.0, (255, 255, 255), 3
#     )

#     def kmeans_color_quantization(image, clusters=5, rounds=10):
#         data = image.reshape((-1, 3))
#         data = np.float32(data)

#         # Define criteria and apply kmeans
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
#         ret, label, center = cv2.kmeans(
#             data, clusters, None, criteria, rounds, cv2.KMEANS_RANDOM_CENTERS
#         )

#         # Convert centers to uint8
#         center = np.uint8(center)
#         res = center[label.flatten()]
#         result_image = res.reshape((image.shape))

#         # Calculate and plot histogram
#         unique, counts = np.unique(label, return_counts=True)
#         sort_idx = np.argsort(counts)[::-1]
#         sorted_colors = center[sort_idx]
#         sorted_counts = counts[sort_idx]
#         labels = []
#         for color in sorted_colors:
#             labels.append(f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}")

#         # # Plot the colors as a pie chart
#         # plt.figure(figsize=(6, 4))
#         # plt.pie(
#         #     sorted_counts,
#         #     labels=labels,
#         #     colors=[color / 255.0 for color in sorted_colors],
#         #     autopct="%1.1f%%",
#         # )
#         # plt.title("Color distribution")
#         # plt.savefig(f"out/color_distribution_{file_path.split('/')[-1]}")
#         # plt.show()
#         for label in labels:
#             if label in center_colors:
#                 center_colors[label] += counts[labels.index(label)]
#             else:
#                 center_colors[label] = counts[labels.index(label)]

#     # kmeans_color_quantization(image, clusters=5, rounds=10)

#     # Combine original and overlay side by side
#     combined = np.hstack((image, overlay))

#     return ratio, combined, image


# # Variables to store results
# ratios = []
# filenames = []

# # Process images
# i = 1
# while True:
#     file_name = f"img/T={i}h.jpg"
#     if not os.path.exists(file_name):
#         break

#     ratio, combined_image, original_image = process_image(file_name)
#     if original_image is None:
#         break

#     # Save combined image for GIF
#     combined_filename = f"out/combined_{i}.jpg"
#     cv2.imwrite(combined_filename, combined_image)
#     filenames.append(combined_filename)

#     # Store the ratio
#     ratios.append(ratio)
#     i += 1

# # Create GIF
# with imageio.get_writer("out/segmentation_side_by_side.gif", mode="I", fps=1) as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)

# # Plotting the graph
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, len(ratios) + 1), ratios, marker="o")
# plt.title("Oil Area Ratio Over Time")
# plt.xlabel("Time (hours)")
# plt.ylabel("Oil to Total Area Ratio")
# plt.grid(True)
# plt.savefig("out/ratio_plot.png")
# # plt.show()

# print(center_colors)

center_colors = {
    "#878874": 441424,
    "#8c928a": 387864,
    "#939a95": 380115,
    "#030303": 147132,
    "#fcfcfc": 23479,
    "#939994": 462856,
    "#878873": 427228,
    "#8b9188": 319335,
    "#8f9690": 798453,
    "#848778": 352174,
    "#9ea59f": 58892,
    "#fdfdfd": 4640,
    "#8d948e": 608787,
    "#848676": 325669,
    "#959c98": 1068700,
    "#8c948e": 1307064,
    "#838576": 306079,
    "#959c99": 290612,
    "#8c938d": 568101,
    "#959b97": 359348,
    "#828374": 281960,
    "#fbfbfb": 14485,
    "#8d948f": 714637,
    "#969d99": 595389,
    "#808276": 211412,
    "#8c938e": 1262163,
    "#7f8279": 183225,
    "#7f8278": 381867,
    "#fbfcfb": 4910,
}

# Plot center colors
plt.figure(figsize=(8, 6))
for hex, count in center_colors.items():
    plt.scatter(0, 0, color=hex, s=count / 1000, label=f"{hex} ({count})")
plt.title("Center Colors")
plt.axis("off")
plt.legend()
plt.show()

# # Convert hex to RGB
# colors_rgb = np.array(
#     [
#         [int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)]
#         for hex in center_colors.keys()
#     ]
# )

# # K-means clustering on these colors
# kmeans = KMeans(n_clusters=5, random_state=0).fit(colors_rgb)
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_

# # Plotting
# plt.figure(figsize=(8, 6))
# for label, color in zip(labels, colors_rgb):
#     plt.scatter(
#         color[0], color[1], color=color / 255.0, s=100, label=f"Cluster {label}"
#     )
# plt.scatter(centers[:, 0], centers[:, 1], s=200, color="black", marker="x")
# plt.title("Clustered Color Centroids")
# plt.xlabel("Red")
# plt.ylabel("Green")
# plt.show()

# # Output the new centroids in hex format
# new_centers_hex = ["#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2])) for c in centers]
# print("New Centers in Hex:", new_centers_hex)


# def process_centroid_image(file_path, color_centroids, threshold=30):
#     # Load the image
#     image = cv2.imread(file_path)
#     if image is None:
#         return None, None

#     # Create a mask based on color proximity
#     mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#     for center in color_centroids:
#         # Calculate the distance from each color centroid
#         distance = np.sqrt(np.sum((image - center) ** 2, axis=2))
#         mask[distance < threshold] = 255

#     # Morphological opening to remove noise
#     kernel = np.ones((5, 5), np.uint8)
#     cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     # Overlay mask on the image
#     segmented = cv2.bitwise_and(image, image, mask=cleaned_mask)

#     # Save the segmented output
#     output_path = file_path.replace("img", "out").replace(".jpg", "_segmented.jpg")
#     cv2.imwrite(output_path, segmented)

#     return segmented, output_path


# # Example usage:
# for i in range(0, 11):
#     file_path = f"img/T={i}h.jpg"
#     segmented_image, output_path = process_centroid_image(file_path, centers, threshold=30)
#     print(f"Segmented image saved to: {output_path}")

# # Optionally display the result using matplotlib
# plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
# plt.title("Segmented Image")
# plt.show()
