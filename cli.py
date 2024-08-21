import numpy as np
import argparse
import imageio
import glob
import cv2
import os

# Constants
OUTPUT_DIMENSIONS = (1024, 1024)
SEARCH_RADIUS = 3
FULL_ROTATION_DEGREES = 360
ROTATION_STEP_DEGREES = 90
MIN_RADIUS_DIVISOR, MAX_RADIUS_DIVISOR = 4, 2
BLUR_KERNEL_SIZE = (9, 9)
BLUR_SIGMA = 2
HOUGH_CIRCLE_SETTINGS = {
    'dp': 1,
    'minDist': 100,
    'param1': 50,
    'param2': 30
}
TINT_OPTIONS = {
    'normal': [0.93, 0.44, 0.01],
    'green': [0.44, 0.93, 0.01],
    'pink': [0.93, 0.01, 0.44],
    'pink2': [0.95, 0.25, 0.85],
    'blue': [0.01, 0.44, 0.93],
    'blue2': [0.31, 0.50, 0.98],
    'blue3': [0.11, 0.50, 0.98],
    'purple': [0.45, 0.11, 0.51]
}

def compute_image_difference(img1, img2):
    """Compute the L2 norm (Euclidean distance) between two images."""
    return cv2.norm(img1, img2, cv2.NORM_L2)

def rotate_image(image, angle):
    """Rotate the image by the specified angle, keeping the original size."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def crop_image(image, center, size):
    """
    Crop the image around a specified center point. If the crop area 
    extends beyond the image boundaries, fill the area with black.
    """
    x, y, _ = center
    height, width = image.shape[:2]
    crop_width, crop_height = size

    # Calculate crop boundaries
    x_min = int(x - crop_width // 2)
    y_min = int(y - crop_height // 2)
    x_max = x_min + crop_width
    y_max = y_min + crop_height

    # Create a black canvas for the desired size
    cropped_image = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)

    # Determine the overlapping region between the original image and the crop area
    src_x_min = max(0, x_min)
    src_y_min = max(0, y_min)
    src_x_max = min(width, x_max)
    src_y_max = min(height, y_max)

    # Calculate the corresponding region in the cropped image
    dst_x_min = src_x_min - x_min
    dst_y_min = src_y_min - y_min
    dst_x_max = dst_x_min + (src_x_max - src_x_min)
    dst_y_max = dst_y_min + (src_y_max - src_y_min)

    # Copy the overlapping region from the original image to the cropped image
    cropped_image[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = image[src_y_min:src_y_max, src_x_min:src_x_max]

    return cropped_image

def find_sun_center(image):
    """Detect the center of the sun in the image using Hough Circle Transform."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, BLUR_SIGMA)
    min_radius = int(min(image.shape[:2]) / MIN_RADIUS_DIVISOR)
    max_radius = int(min(image.shape[:2]) / MAX_RADIUS_DIVISOR)

    circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, minRadius=min_radius,
        maxRadius=max_radius, **HOUGH_CIRCLE_SETTINGS
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))[0]
        return min(circles, key=lambda c: (image.shape[1] // 2 - c[0])**2 + (image.shape[0] // 2 - c[1])**2)[:3]
    
    return None

def enhance_image(image, brightness, contrast_factor, tint='blue'):
    """Apply color enhancement, tinting, and contrast adjustments to the image."""
    # Adjust brightness
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v = np.clip(v + brightness, 0, 255)
    image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    # Apply tinting
    image_float = image.astype(np.float32) / 255.0
    tint_color = np.array(TINT_OPTIONS[tint])
    tinted_image = image_float * tint_color

    # Adjust contrast
    mean_value = np.mean(tinted_image, axis=(0, 1))
    tinted_image = np.clip((tinted_image - mean_value) * contrast_factor + mean_value, 0, 1)
    tinted_image = (tinted_image * 255).astype(np.uint8)

    return cv2.normalize(tinted_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def process_images(input_dir, output_gif, calibration_image_path, frame_duration, tint, brightness, contrast_factor):
    """Process all images in the input directory and save the output as a GIF."""
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    gif_frames = []

    # Process the calibration image
    calibration_image = cv2.imread(calibration_image_path)
    calibration_center = find_sun_center(calibration_image)
    if calibration_center is None:
        return
    
    calibration_crop = crop_image(calibration_image, calibration_center, OUTPUT_DIMENSIONS)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        sun_center = find_sun_center(image)
        if sun_center is None:
            print(f"No circles detected on {image_path}, skipping")
            continue

        # Find best translation
        translations = [
            (tx, ty, compute_image_difference(calibration_crop, crop_image(image, [tx, ty, sun_center[2]], OUTPUT_DIMENSIONS)))
            for tx in range(sun_center[0] - SEARCH_RADIUS, sun_center[0] + SEARCH_RADIUS)
            for ty in range(sun_center[1] - SEARCH_RADIUS, sun_center[1] + SEARCH_RADIUS)
        ]
        best_tx, best_ty, _ = min(translations, key=lambda x: x[2])
        cropped_image = crop_image(image, [best_tx, best_ty, sun_center[2]], OUTPUT_DIMENSIONS)

        # Find best rotation
        rotations = [
            (angle, compute_image_difference(calibration_crop, rotate_image(cropped_image, angle)))
            for angle in range(0, FULL_ROTATION_DEGREES, ROTATION_STEP_DEGREES)
        ]
        best_rotation_angle, _ = min(rotations, key=lambda x: x[1])
        rotated_image = rotate_image(cropped_image, best_rotation_angle)

        enhanced_image = enhance_image(rotated_image, brightness=brightness, contrast_factor=contrast_factor, tint=tint)
        gif_frames.append(enhanced_image)

    # Save the processed frames as a GIF
    imageio.mimsave(output_gif, gif_frames, loop=65535, duration=frame_duration)

def main():
    parser = argparse.ArgumentParser(description="Process sun images and create a GIF.")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("--output_file", default="proba2.gif", help="Output GIF/WEBP filename (default: proba2.gif)")
    parser.add_argument("--calibration_image", default=".calibration.png", help="Calibration image filename (default: .calibration.png)")
    parser.add_argument("--frame_duration", type=int, default=100, help="Frame duration in milliseconds (default: 100)")
    parser.add_argument("--tint", choices=TINT_OPTIONS.keys(), default="normal", help="Color tint to apply (default: normal)")
    parser.add_argument("--brightness", type=int, default=20, help="Brightness modifier (default: 20)")
    parser.add_argument("--contrast_factor", type=int, default=4, help="Contrast factor (default: 4)")
    args = parser.parse_args()
    process_images(args.input_dir, args.output_file, args.calibration_image, args.frame_duration, args.tint, args.brightness, args.contrast_factor)
    print(f"Output written to: {args.output_file}")

if __name__ == "__main__":
    main()
