# equalization.py
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def manual_equalize_histogram(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = np.zeros(256, dtype=int)
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    cdf = hist.cumsum()
    cdf_min = cdf.min()
    if cdf_min == cdf.max():
        return image
    cdf_normalized = ((cdf - cdf_min) * 255 / (cdf.max() - cdf_min)).astype(np.uint8)
    equalized_img = cdf_normalized[image]
    return equalized_img

def process_single_image_equalization(filename, input_dir, output_dir):
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        return
    equalized = manual_equalize_histogram(img)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, filename), equalized)

def process_dataset_equalization(input_root, output_root, max_images_per_class=None):
    classes = ["Original", "Forged"]
    for label in classes:
        input_dir = os.path.join(input_root, label)
        output_dir = os.path.join(output_root, label)
        os.makedirs(output_dir, exist_ok=True)
        filenames = sorted(os.listdir(input_dir))
        if max_images_per_class is not None:
            filenames = filenames[:max_images_per_class]
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda f: process_single_image_equalization(f, input_dir, output_dir), filenames), total=len(filenames)))
        print(f"Done histogram equalization for {label}.")
