import pandas as pd
from typing import Dict
import os
import numpy as np
from Scripts.IGTD_Functions import (
    min_max_transform,
    table_to_image,
    drop_numerical_outliers,
)

from dataloaders.dataloader import *
from factory import (
    create_data_loader,
)

# Create an instance of the specified data loader class
dataset_name = "bufix"
data_loader = create_data_loader(
    dataset_name,
    test_size=0.2,
    normalize_features="mean_std",
    encode_categorical=True,
    return_extra_info=False,
    random_state=4200,
)
X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()

exclude_cols = []

num_row = 10  # Number of pixel rows in image representation
num_col = 11  # Number of pixel columns in image representation
num = (
    num_row * num_col
)  # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = (
    3  # Size of pictures (in inches) saved during the execution of IGTD algorithm.
)

X_train = min_max_transform(X_train, feature_range=(0, 255), exclude_cols=exclude_cols)
X_test = min_max_transform(X_test, feature_range=(0, 255), exclude_cols=exclude_cols)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = "Euclidean"
image_dist_method = "Euclidean"
error = "abs"
result_dir = f"/Users/mdicio/Documents/GitHub/RealWTab/modelsdefinition/IGTD/results/{dataset_name}_igtd_{fea_dist_method}_{image_dist_method}"

max_step = 1000000  # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 10000  # The number of iterations for determining algorithm convergence. If the error reduction rate
# is smaller than a pre-set threshold for val_step itertions, the algorithm converges.
min_gain = 0.01

os.makedirs(name=result_dir, exist_ok=True)
table_to_image(
    X_train,
    [num_row, num_col],
    fea_dist_method,
    image_dist_method,
    save_image_size,
    max_step,
    val_step,
    result_dir,
    error,
    min_gain=min_gain,
    save_mode="bulk",
    exclude_cols=exclude_cols,
)

# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
fea_dist_method = "Pearson"
image_dist_method = "Manhattan"
error = "squared"
result_dir = f"/Users/mdicio/Documents/GitHub/RealWTab/modelsdefinition/IGTD/results/{dataset_name}_igtd_{fea_dist_method}_{image_dist_method}"
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(
    X_train,
    [num_row, num_col],
    fea_dist_method,
    image_dist_method,
    save_image_size,
    max_step,
    val_step,
    result_dir,
    error,
    min_gain=min_gain,
)
