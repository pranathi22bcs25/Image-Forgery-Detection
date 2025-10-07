# main.py
from resize import process_dataset_resize
from filtering import process_dataset_filter
from equalization import process_dataset_equalization

# ---------------------------
# Step 1: Resize images
# ---------------------------
input_dataset_path = "/kaggle/input/image-forgery-detection/Dataset"

nearest_resized_path = "/content/Resized/nearest"
bilinear_resized_path = "/content/Resized/bilinear"

# Resizing nn
process_dataset_resize(
    input_dataset_path,
    nearest_resized_path,
    method="nearest"
)

# Resizing bilinear
process_dataset_resize(
    input_dataset_path,
    bilinear_resized_path,
    method="bilinear"
)

# ---------------------------
# Step 2: Gaussian Filtering
# ---------------------------
gaussian_filtered_input = bilinear_resized_path
gaussian_filtered_output = "/content/Filtered_All/gaussian"

process_dataset_filter(
    gaussian_filtered_input,
    gaussian_filtered_output,
    filter_name="gaussian"
)

# ---------------------------
# Step 3: Histogram Equalization
# ---------------------------
equalization_input = gaussian_filtered_output
equalized_output = "/content/Equalized_Dataset"

process_dataset_equalization(
    equalization_input,
    equalized_output
)
