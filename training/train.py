import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.pointcloud_data import PointCloudData
from models.generator import Generator
from models.discriminator import Discriminator
from utils.point_operations import adversarial_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    PointSampler(1024),
    Normalize(),
    RandRotation_z(),
    RandomNoise(),
    ToTensor()
])

path = "/path/to/your/Bench/dataset"
train_ds = PointCloudData(path, transform=train_transforms)
train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)

# Models
latent_dim = 128
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        real_point_clouds = data['pointcloud'].float().to(device)
        batch_size = real_point_clouds.size(0)

        valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
        fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_point_clouds), valid)

        z = torch.randn((batch_size, latent_dim)).to(device)
        generated_point_clouds = generator(z)
        fake_loss = adversarial_loss(discriminator(generated_point_clouds.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        g_loss = adversarial_loss(discriminator(generated_point_clouds), valid)
        g_loss.backward()
        optimizer_G.step()

        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    # Save model every few epochs
    torch.save(generator.state_dict(), f'../checkpoints/generator_epoch_{epoch}.pth')
