import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataprocessor import RegularDataProcessor


class WGAN_GP(nn.Module):
    def __init__(
        self, model_parameters, n_generator=1, n_critic=1, gradient_penalty_weight=10
    ):
        super().__init__()
        self.n_critic = n_critic
        self.n_generator = n_generator
        self.gradient_penalty_weight = gradient_penalty_weight
        self.noise_dim = model_parameters["noise_dim"]
        self.layers_dim = model_parameters["layers_dim"]
        self.batch_size = model_parameters["batch_size"]
        self.g_lr = model_parameters["g_lr"]
        self.d_lr = model_parameters["d_lr"]
        self.beta_1 = model_parameters["beta_1"]
        self.beta_2 = model_parameters["beta_2"]
        self.lrelu_alpha = model_parameters.get("lrelu_alpha", 0.2)
        self.dropout_pct = model_parameters.get("dropout_pct", 0.3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def define_gan_optimizers(self):
        g_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.g_lr, betas=(self.beta_1, self.beta_2)
        )
        c_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.d_lr, betas=(self.beta_1, self.beta_2)
        )
        return g_optimizer, c_optimizer

    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device)
        x_hat = epsilon * real + (1 - epsilon) * fake
        # x_hat.requires_grad = True
        d_hat = self.critic(x_hat)
        gradients = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_gradients(self, real, g_optimizer, c_optimizer):
        for _ in range(self.n_critic):
            critic_loss = self.c_lossfn(real)
            c_optimizer.zero_grad()
            critic_loss.backward()
            c_optimizer.step()

        gen_loss = None
        for _ in range(self.n_generator):
            gen_loss = self.g_lossfn(real)
            g_optimizer.zero_grad()
            gen_loss.backward()
            g_optimizer.step()

        return critic_loss.item(), gen_loss.item() if gen_loss else None

    def c_lossfn(self, real):
        noise = torch.randn(real.shape[0], self.generator.noise_dim, device=self.device)
        fake = self.generator(noise).to(self.device)
        logits_real = self.critic(real).to(self.device)
        logits_fake = self.critic(fake).to(self.device)
        gp = self.gradient_penalty(real, fake)
        c_loss = (
            torch.mean(logits_fake)
            - torch.mean(logits_real)
            + gp * self.gradient_penalty_weight
        )
        # TODO TESTING ABSOLUTE LOSS

        # return c_loss
        return torch.abs(c_loss)

    def g_lossfn(self, real):
        noise = torch.randn(real.size(0), self.generator.noise_dim, device=self.device)
        fake = self.generator(noise).to(self.device)
        logits_fake = self.critic(fake).to(self.device)
        g_loss = -torch.mean(logits_fake)
        return g_loss

    def get_data_batch(self, train_data, batch_size, seed=0):
        start_i = (batch_size * seed) % len(train_data)
        stop_i = start_i + batch_size
        train_ix = np.random.choice(
            len(train_data), replace=False, size=len(train_data)
        )
        train_ix = np.concatenate(
            [train_ix, train_ix]
        )  # duplicate to cover ranges past the end of the set
        return torch.tensor(
            train_data[train_ix[start_i:stop_i]], device=self.device
        ).float()

    def train_step(self, train_data, optimizers):
        cri_loss, gen_loss = self.update_gradients(train_data, *optimizers)
        return cri_loss, gen_loss

    def fit(self, data, train_arguments, num_cols, cat_cols):
        self.data_processor = RegularDataProcessor(num_cols, cat_cols)

        self.data_processor.fit(data)
        processed_data = self.data_processor.process_data(data)

        self.data_dim = processed_data.shape[1]

        self.generator = Generator(
            self.noise_dim,
            self.layers_dim,
            self.data_dim,
            self.lrelu_alpha,
            self.dropout_pct,
        ).to(self.device)
        self.critic = Critic(self.data_dim, self.dropout_pct).to(self.device)

        optimizers = self.define_gan_optimizers()

        iterations = int(abs(data.shape[0] / self.batch_size) + 1)

        with tqdm(
            range(train_arguments.epochs), desc="Epochs", position=0, leave=True
        ) as pbar:
            for epoch in pbar:
                for _ in range(iterations):
                    batch_data = self.get_data_batch(
                        processed_data, self.batch_size
                    ).float()
                    cri_loss, ge_loss = self.train_step(batch_data, optimizers)

                if epoch % train_arguments.sample_interval == 0:
                    # Test here data generation step
                    # save model checkpoints
                    if not os.path.exists("./cache"):
                        os.mkdir("./cache")
                    model_checkpoint_base_name = (
                        "./cache/"
                        + train_arguments.cache_prefix
                        + "_{}_model_weights_step_{}.pth"
                    )
                    torch.save(
                        self.generator.state_dict(),
                        model_checkpoint_base_name.format("generator", epoch),
                    )
                    torch.save(
                        self.critic.state_dict(),
                        model_checkpoint_base_name.format("critic", epoch),
                    )
                pbar.set_postfix(
                    {
                        "Epoch": f"{epoch+1}/{train_arguments.epochs}",
                        "Generator Loss": f"{ge_loss}",
                        "Critic Loss": f"{cri_loss}",
                    }
                )
            pbar.close()

    def _sample_unscaled(self, num_samples):
        """Generate synthetic data samples using the trained generator.

        Args:
            num_samples (int): Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: DataFrame containing the synthetic data samples.
        """
        noise_input = torch.randn(num_samples, self.noise_dim).to(self.device)
        generated_data = self.generator(noise_input).detach().cpu().numpy()
        col_names = self.data_processor._col_order_
        # Convert generated data to a DataFrame
        generated_df = pd.DataFrame(generated_data, columns=col_names)

        return generated_df

    def sample(self, num_samples):
        """Generate synthetic data samples using the trained generator.

        Args:
            num_samples (int): Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: DataFrame containing the synthetic data samples.
        """
        generated_df = self._sample_unscaled(num_samples)

        means = self.data_processor._num_mean
        stds = self.data_processor._num_std
        for col, mean, std in zip(generated_df.columns, means, stds):
            generated_df[col] = generated_df[col] * std + mean

        return generated_df


class Generator(nn.Module):
    def __init__(self, noise_dim, layers_dim, data_dim, lrelu_alpha, dropout_pct):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim
        self.data_dim = data_dim
        self.model = nn.Sequential(
            nn.Linear(noise_dim, layers_dim),
            nn.LeakyReLU(lrelu_alpha),
            nn.Dropout(dropout_pct),
            nn.Linear(layers_dim, layers_dim * 2),
            nn.LeakyReLU(lrelu_alpha),
            nn.Dropout(dropout_pct),
            nn.Linear(layers_dim * 2, layers_dim * 4),
            nn.LeakyReLU(lrelu_alpha),
            nn.Dropout(dropout_pct),
            nn.Linear(layers_dim * 4, data_dim),
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, data_dim, lrelu_alpha):
        super(Critic, self).__init__()
        self.data_dim = data_dim
        self.model = nn.Sequential(
            nn.Linear(data_dim, data_dim * 4),
            nn.LeakyReLU(lrelu_alpha),
            nn.Linear(data_dim * 4, data_dim * 2),
            nn.LeakyReLU(lrelu_alpha),
            nn.Linear(data_dim * 2, data_dim),
            nn.LeakyReLU(lrelu_alpha),
            nn.Linear(data_dim, 1),
        )

    def forward(self, x):
        return self.model(x)
