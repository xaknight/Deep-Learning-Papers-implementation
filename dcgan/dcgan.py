import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
LEARNING_RATE = 2e-4  # Original paper used 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64  # Resize CIFAR images to 64x64 (common for GANs)
CHANNELS_IMG = 3  # CIFAR images are color images (RGB)
NOISE_DIM = 100  # Dimension of the noise vector z
NUM_EPOCHS = 50
FEATURES_DISC = 64  # Feature size for Discriminator
FEATURES_GEN = 64   # Feature size for Generator

# Transformations for CIFAR-10 dataset
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# CIFAR-10 Dataset and DataLoader
dataset = datasets.CIFAR10(root="dataset/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator Network
class Generator(nn.Module):
    def __init__(self, noise_dim, channels_img, features_gen):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x noise_dim x 1 x 1
            self._block(noise_dim, features_gen * 16, 4, 1, 0),  # Output: (features_gen*16) x 4 x 4
            self._block(features_gen * 16, features_gen * 8, 4, 2, 1),  # Output: (features_gen*8) x 8 x 8
            self._block(features_gen * 8, features_gen * 4, 4, 2, 1),  # Output: (features_gen*4) x 16 x 16
            self._block(features_gen * 4, features_gen * 2, 4, 2, 1),  # Output: (features_gen*2) x 32 x 32
            nn.ConvTranspose2d(
                features_gen * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: channels_img x 64 x 64
            nn.Tanh(), # Output in range [-1, 1] to match normalization
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False, # No bias in BatchNorm layers
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.net(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_disc):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: channels_img x 64 x 64
            nn.Conv2d(channels_img, features_disc, kernel_size=4, stride=2, padding=1), # Output: features_disc x 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_disc, features_disc * 2, 4, 2, 1), # Output: (features_disc*2) x 16 x 16
            self._block(features_disc * 2, features_disc * 4, 4, 2, 1), # Output: (features_disc*4) x 8 x 8
            self._block(features_disc * 4, features_disc * 8, 4, 2, 1), # Output: (features_disc*8) x 4 x 4
            nn.Conv2d(features_disc * 8, 1, kernel_size=4, stride=2, padding=0), # Output: 1 x 1 x 1
            nn.Sigmoid(), # Output probability [0, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False, # No bias in BatchNorm layers
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2), # LeakyReLU for Discriminator (common practice)
        )

    def forward(self, x):
        return self.disc(x).reshape(-1, 1) # Flatten to (N, 1)

# Initialize networks and optimizers
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen.to(device)
disc.to(device)

optimizer_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)) # Beta1=0.5 is common in GANs
optimizer_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Loss function (Binary Cross Entropy Loss)
criterion = nn.BCELoss()

# Fixed noise for visualization during training
fixed_noise = torch.randn((32, NOISE_DIM, 1, 1)).to(device)
real_label = 1
fake_label = 0

# Training Loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        # Discriminator on real data
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.full((batch_size,), real_label, dtype=torch.float).to(device))

        # Discriminator on fake data
        disc_fake = disc(fake.detach()).reshape(-1) # Detach to avoid backprop through Generator when training Discriminator
        loss_disc_fake = criterion(disc_fake, torch.full((batch_size,), fake_label, dtype=torch.float).to(device))

        # Total Discriminator loss
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        # Backpropagation and optimization for Discriminator
        disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        ### Train Generator: minimize log(1 - D(G(z)))  OR maximize log(D(G(z))) - using the modified objective
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.full((batch_size,), real_label, dtype=torch.float).to(device)) # Generator wants Discriminator to think fake images are real

        # Backpropagation and optimization for Generator
        gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # Print progress
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

    # Visualize generated images after each epoch
    with torch.no_grad():
        fake = gen(fixed_noise).detach().cpu()
        img_grid = torchvision.utils.make_grid(fake, normalize=True)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.title(f"Epoch {epoch}")
        plt.axis('off')
        plt.show()