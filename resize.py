# resize.py
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def resize_nearest(img, new_width, new_height):
    height, width = img.shape[:2]
    resized = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for y in range(new_height):
        for x in range(new_width):
            orig_x = int(x * width / new_width)
            orig_y = int(y * height / new_height)
            resized[y, x] = img[orig_y, orig_x]
    return resized

def resize_bilinear(img, new_width, new_height):
    height, width = img.shape[:2]
    resized = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for y in range(new_height):
        for x in range(new_width):
            gx = (x + 0.5) * width / new_width
            gy = (y + 0.5) * height / new_height
            x0 = int(np.floor(gx))
            x1 = min(x0 + 1, width - 1)
            y0 = int(np.floor(gy))
            y1 = min(y0 + 1, height - 1)
            dx = gx - x0
            dy = gy - y0
            for c in range(3):
                val = (1 - dx) * (1 - dy) * img[y0, x0, c] + \
                      dx * (1 - dy) * img[y0, x1, c] + \
                      (1 - dx) * dy * img[y1, x0, c] + \
                      dx * dy * img[y1, x1, c]
                resized[y, x, c] = int(val)
    return resized

def process_single_image_resize(args):
    img_path, resize_fn, size, output_dir = args
    filename = os.path.basename(img_path)
    try:
        img = cv2.imread(img_path)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = resize_fn(img, size[0], size[1])
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(save_path, resized_bgr)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

def process_dataset_resize(input_root, output_root, method="nearest", size=(256,256), max_images_per_class=None):
    resize_fn = resize_nearest if method=="nearest" else resize_bilinear
    classes = ["Original", "Forged"]
    for label in classes:
        input_dir = os.path.join(input_root, label)
        output_dir = os.path.join(output_root, method, label)
        os.makedirs(output_dir, exist_ok=True)
        filenames = sorted(os.listdir(input_dir))
        if max_images_per_class is not None:
            filenames = filenames[:max_images_per_class]
        tasks = [(os.path.join(input_dir, f), resize_fn, size, output_dir) for f in filenames]
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(process_single_image_resize, tasks), total=len(tasks)))
        print(f"Done with {label} using {method} resizing.")
