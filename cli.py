import cv2
import numpy as np
import glob
import imageio
import argparse
import os

# Constants
OUTPUT_SIZE = (1024, 1024)
GRID_SEARCH_RANGE = 4
ROTATION_RANGE = 360
ROTATION_STEP = 90
MIN_RADIUS_FACTOR, MAX_RADIUS_FACTOR = 4, 2
GAUSSIAN_BLUR_KERNEL = (9, 9)
GAUSSIAN_BLUR_SIGMA = 2
HOUGH_CIRCLE_PARAMS = {
    'dp': 1,
    'minDist': 100,
    'param1': 50,
    'param2': 30
}
TINT_COLORS = {
    'normal': [0.93, 0.44, 0.01],
    'green': [0.44, 0.93, 0.01],
    'pink': [0.93, 0.01, 0.44],
    'blue': [0.01, 0.44, 0.93]
}

def calculate_image_difference(img1, img2):
    """Calculate the L1 norm difference between two images."""
    return cv2.norm(img1, img2, cv2.NORM_L1)

def rotate_image(image, angle):
    """Rotate the image by the specified angle, maintaining original size."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def crop_image(img, center, size):
    """
    Crop the image around a specified center point.
    If the crop area extends beyond the image boundaries, fill with black.
    """
    x, y, _ = center
    height, width = img.shape[:2]
    crop_width, crop_height = size

    # Calculate crop boundaries
    x_min = int(x - crop_width // 2)
    y_min = int(y - crop_height // 2)
    x_max = x_min + crop_width
    y_max = y_min + crop_height

    # Create a black canvas of the desired size
    cropped = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)

    # Calculate the overlapping region between the original image and the crop area
    src_x_min = max(0, x_min)
    src_y_min = max(0, y_min)
    src_x_max = min(width, x_max)
    src_y_max = min(height, y_max)

    # Calculate the corresponding region in the output image
    dst_x_min = src_x_min - x_min
    dst_y_min = src_y_min - y_min
    dst_x_max = dst_x_min + (src_x_max - src_x_min)
    dst_y_max = dst_y_min + (src_y_max - src_y_min)

    # Copy the overlapping region
    cropped[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = img[src_y_min:src_y_max, src_x_min:src_x_max]

    return cropped

def find_sun_center(img):
    """Detect the center of the sun in the image using Hough Circle Transform."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA)
    min_radius = int(min(img.shape[:2]) / MIN_RADIUS_FACTOR)
    max_radius = int(min(img.shape[:2]) / MAX_RADIUS_FACTOR)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, minRadius=min_radius,
        maxRadius=max_radius, **HOUGH_CIRCLE_PARAMS
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))[0]
        return min(circles, key=lambda c: (img.shape[1]//2 - c[0])**2 + (img.shape[0]//2 - c[1])**2)[:3]
    print("No circles detected.")
    return None

def enhance_image(image, brightness, contrast_factor, tint='blue'):
    """Apply color enhancement, tinting, and contrast adjustments to the image."""
    # Adjust brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + brightness, 0, 255)
    image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    # Apply tinting
    image_float = image.astype(np.float32) / 255.0
    tint_color = np.array(TINT_COLORS[tint])
    tinted = image_float * tint_color

    # Adjust contrast
    mean = np.mean(tinted, axis=(0, 1))
    tinted = np.clip((tinted - mean) * contrast_factor + mean, 0, 1)
    tinted = (tinted * 255).astype(np.uint8)

    return cv2.normalize(tinted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def process_images(input_dir, output_gif, calibration_image, frame_duration, tint, brightness, contrast_factor):
    """Process all images in the input directory and save as a GIF."""
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    frames = []

    # Process calibration image
    calibration_img = cv2.imread(calibration_image)
    calibration_center = find_sun_center(calibration_img)
    calibration_crop = crop_image(calibration_img, calibration_center, OUTPUT_SIZE)
    calibration_crop = enhance_image(calibration_crop, brightness=brightness, contrast_factor=contrast_factor, tint=tint)

    for filename in image_files:
        img = cv2.imread(filename)
        center = find_sun_center(img)

        # Find best translation
        translations = [
            (tx, ty, calculate_image_difference(calibration_crop, crop_image(img, [tx, ty, center[2]], OUTPUT_SIZE)))
            for tx in range(center[0] - GRID_SEARCH_RANGE, center[0] + GRID_SEARCH_RANGE)
            for ty in range(center[1] - GRID_SEARCH_RANGE, center[1] + GRID_SEARCH_RANGE)
        ]
        best_tx, best_ty, _ = min(translations, key=lambda x: x[2])
        crop = crop_image(img, [best_tx, best_ty, center[2]], OUTPUT_SIZE)

        # Find best rotation
        rotations = [
            (angle, calculate_image_difference(calibration_crop, rotate_image(crop, angle)))
            for angle in range(-1, ROTATION_RANGE, ROTATION_STEP)
        ]
        best_angle, _ = min(rotations, key=lambda x: x[1])
        rotated = rotate_image(crop, best_angle)

        enhanced = enhance_image(rotated, brightness=brightness, contrast_factor=contrast_factor, tint=tint)
        frames.append(enhanced)

    # Save as GIF
    imageio.mimsave(output_gif, frames, loop=65535 duration=frame_duration)

def main():
    parser = argparse.ArgumentParser(description="Process sun images and create a GIF.")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("--output_file", default="proba2.gif", help="Output GIF/WEBP filename (default: proba2.gif)")
    parser.add_argument("--calibration_image", default=".calibration.png", help="Calibration image filename (default: .calibration.png)")
    parser.add_argument("--frame_duration", type=int, default=100, help="Frame duration in milliseconds (default: 100)")
    parser.add_argument("--tint", choices=TINT_COLORS.keys(), default="normal", help="Color tint to apply (default: normal)")
    parser.add_argument("--brightness", type=int, default=20, help="Brightness modifier (default: 20)")
    parser.add_argument("--contrast_factor", type=int, default=4, help="Contrast factor (default: 4)")
    args = parser.parse_args()

    process_images(args.input_dir, args.output_file, args.calibration_image, args.frame_duration, args.tint, args.brightness, args.contrast_factor)
    print(f"output written to: {args.output_file}")

if __name__ == "__main__":
    main()
