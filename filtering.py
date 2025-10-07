# filtering.py
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def mean_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='constant', constant_values=0)
    filtered_img = np.zeros_like(image, dtype=float)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            window = padded_img[i:i+kernel_size, j:j+kernel_size]
            filtered_img[i, j] = np.mean(window)
    return filtered_img.astype(np.uint8)

def median_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    filtered = np.zeros_like(image, dtype=float)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.median(region)
    return filtered.astype(np.uint8)

def gaussian_kernel(size=3, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def gaussian_filter(image, kernel_size=3, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    filtered = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.sum(region * kernel)
    return filtered.astype(np.uint8)

def process_single_image_filter(filename, input_dir, output_dir, filter_fn):
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    filtered = filter_fn(img)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, filename), filtered)

def process_dataset_filter(input_root, output_root, filter_name="gaussian", kernel_size=3, max_images_per_class=None):
    filter_fn = mean_filter if filter_name=="mean" else median_filter if filter_name=="median" else lambda img: gaussian_filter(img, kernel_size)
    classes = ["Original", "Forged"]
    for label in classes:
        input_dir = os.path.join(input_root, label)
        output_dir = os.path.join(output_root, filter_name, label)
        os.makedirs(output_dir, exist_ok=True)
        filenames = sorted(os.listdir(input_dir))
        if max_images_per_class is not None:
            filenames = filenames[:max_images_per_class]
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda f: process_single_image_filter(f, input_dir, output_dir, filter_fn), filenames), total=len(filenames)))
        print(f"Done filtering {label} with {filter_name}.")
