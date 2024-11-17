import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# from model import WGAN_GP

# Define the WGAN_GP class and related Generator and Critic classes

# Define the model parameters
model_parameters = {
    "batch_size": 64,
    "noise_dim": 100,
    "layers_dim": 128,
    "g_lr": 0.0001,
    "d_lr": 0.0001,
    "beta_1": 0.5,
    "beta_2": 0.999,
}

# Instantiate the WGAN_GP model
wgan_gp_model = WGAN_GP(model_parameters)

# Define training arguments
class TrainParameters:
    def __init__(self, epochs=100, sample_interval=10, cache_prefix="wgan"):
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.cache_prefix = cache_prefix

# Test the processor
# Create a sample DataFrame
data = {
    "numeric1": [1, 2, 3],
    "numeric2": [4, 5, 6],
}
df = pd.DataFrame(data)
print(df.shape)
# Define the numerical and categorical columns
num_cols = ["numeric1", "numeric2"]
cat_cols = []

# Initialize and fit the processor
processor = RegularDataProcessor(num_cols=num_cols, cat_cols=cat_cols)
processor.fit(df)

# Transform data
transformed_data = processor.transform(df)
print(transformed_data)
print(transformed_data.shape)


train_arguments = TrainParameters(epochs=100, sample_interval=10, cache_prefix="wgan")

# Define numerical and categorical columns (not used in this example)
num_cols = df.columns
cat_cols = []

# Train the WGAN_GP model
wgan_gp_model.fit(df, train_arguments, num_cols, cat_cols)

# Generate synthetic data using the trained model
num_synthetic_samples = 1000  # Number of synthetic samples to generate
noise_input = torch.randn(num_synthetic_samples, model_parameters["noise_dim"])
generated_data = wgan_gp_model.generator(noise_input).detach().numpy()

# Convert generated data to a DataFrame
generated_df = pd.DataFrame(
    generated_data, columns=[f"feat_{i}" for i in range(num_features)]
)

# Print the generated DataFrame
print(generated_df.head())
